import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# from stgcn import STGCN
from stgcn_test import STGCN
# from stgcn_origin import STGCN
# from gcn import STGCN
# from TGCN import STGCN
# from GRU import STGCN
# from ASTGCN import STGCN
# from STDN import STGCN
# from LSTM import STGCN

from utils import get_normalized_adj, CalConfusionMatrix
from data_load import Data_load
from logs.logger import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
num_timesteps_input = 12
num_timesteps_output = 3

epochs = 1000
batch_size = 16
N = 249

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
parser.add_argument('--log_file', type=str, default='run_log')
args = parser.parse_args()

logger = Logger(args.log_file)


def train_epoch(training_input, training_target, nodes, batch_size, means, stds):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    loss_mean = 0.0
    for i in range(0, training_input.shape[0] - 64, batch_size):
        net.train()
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        optimizer_lc.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()

            stds = torch.tensor(stds).cuda()
            means = torch.tensor(means).cuda()
            # max_value = torch.tensor(max_value).cuda()

        out = net(A_wave, X_batch, 'train')

        '''
        loss0 是整个网络最终的输出与labels的损失
        loss1 是局部网络的输出与labels的损失
        loss 是loss0与loss1之间的损失
        应该用loss更新局部网络，用loss1 更新前半部分网络，用loss0更新后半部分网络。
        '''

        nodes = torch.LongTensor(nodes)
        loss0 = F.nll_loss(out[0], nodes)
        loss1 = F.nll_loss(out[1], nodes)
        # print("=" * 50)
        # print(loss0, loss1)
        # print(type(loss0), type(loss1))
        loss = F.l1_loss(loss0, loss1)

        # 更新前半部分网络 ( block1 )
        # 设置 requires_grad, 只更新block1的参数
        for name, param in net.named_parameters():
            if "block1" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        loss1.backward(retain_graph=True)
        optimizer1.step()

        # 更新后半部分网络 ( block2, last_temporal, fully_train)
        # 设置 requires_grad ,不更新block1的参数
        for name, param in net.named_parameters():
            if "block1" in name:
                param.requires_grad = False
        loss0.backward(retain_graph=True)
        optimizer0.step()

        # 更新局部网络，已经在optimizer中设置了只更新LC_block，但也会计算其余部分的梯度
        loss.backward()
        optimizer_lc.step()

        epoch_training_losses.append(loss0.detach().cpu().numpy())
        loss_mean = sum(epoch_training_losses)/len(epoch_training_losses)
    return loss_mean


if __name__ == '__main__':
    torch.manual_seed(7)

    # A, max_value, training_input, training_target, val_input, val_target, test_input, test_target = Data_load(num_timesteps_input, num_timesteps_output)

    A, means, stds, training_input, training_target, val_input, val_target, test_input, test_target, nodes = Data_load(num_timesteps_input, num_timesteps_output)

    torch.cuda.empty_cache()    # free cuda memory
    #
    # nodes = torch.Tensor(np.tile(nodes, batch_size).reshape(batch_size, N))

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    if torch.cuda.is_available():
        A_wave = A_wave.cuda()
        # nodes = torch.Tensor(nodes).type(torch.LongTensor).cuda()
        nodes = torch.LongTensor(nodes).cuda()

    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output)
    if torch.cuda.is_available():
        net.cuda()

    optimizer0 = torch.optim.Adam(net.parameters(), lr=1e-3)
    optimizer1 = torch.optim.Adam(net.block1.parameters(), lr=1e-3)     # 只更新block1
    optimizer_lc = torch.optim.Adam(net.LC_block.parameters(), lr=1e-4)  # 用于更新局部网络

    training_losses = []
    validation_losses = []
    validation_MAE = []
    validation_RMSE = []
    validation_MAPE = []
    arr = [3, 6, 9]
    confusion_matrix = torch.zeros(4, 4)
    accuracy = 0
    eval_loss = 0
    max_accuracy, max_precise, max_recall, max_f1_score = 0, 0, 0, 0
    for epoch in range(epochs):
        print("epoch: {}/{}".format(epoch + 1, epochs))
        loss = train_epoch(training_input, training_target, nodes, batch_size=batch_size, means=means, stds=stds)
        training_losses.append(loss)
        torch.cuda.empty_cache()  # free cuda memory
        # Run validation
        with torch.no_grad():
            net.eval()
            if torch.cuda.is_available():
                val_input = val_input.cuda()
                val_target = val_target.cuda()
            out = net(A_wave, val_input, 'eval')

            eval_loss = F.nll_loss(out[0], nodes, size_average=False).to(device="cpu")
            pred = out[0].data.max(1, keepdim=True)[1]
            pred = torch.squeeze(pred)
            for i in range(len(pred)):
                confusion_matrix[nodes[i]][pred[i]] += 1

            accuracy = pred.eq(nodes.data.view_as(pred)).cpu().sum()
            precise, recall, f1_score = CalConfusionMatrix(confusion_matrix)


            accuracy_value = accuracy.item() / len(nodes)
            if accuracy_value > max_accuracy:
                max_accuracy = accuracy_value
            if precise > max_precise:
                max_precise = precise
            if recall > max_recall:
                max_recall = recall
            if f1_score > max_f1_score:
                max_f1_score = f1_score

            print('Evalution: Average loss: {:.4f}'.format(eval_loss))
            print(
                'Current Evalution: Accuracy: {}/{} ({:.4f}), Precise: {:.4f}, ReCall: {:.4f}, F1 Score: {:.4f}'.format(
                    accuracy, len(nodes), accuracy_value, precise, recall, f1_score))
            print(
                'Max Evalution: Accuracy: {:.4f}, Precise: {:.4f}, ReCall: {:.4f}, F1 Score: {:.4f}'.format(
                    max_accuracy, max_precise, max_recall, max_f1_score))


            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

