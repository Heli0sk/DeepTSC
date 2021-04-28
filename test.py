import numpy as np
import torch
import os
from utils import generate_dataset, load_metr_la_data
from data_load import Data_load


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())

# node_38 = np.load("data/nodes_38/node_class.npy", allow_pickle=True)[:, 1]
# node_249 = np.load("data/node_class.npy", allow_pickle=True)[:, 2]
#
# print(node_38.size)
# print("=" * 50)
# print(node_249)

X = np.load("data/flow_249.npy")
# X = np.reshape(X, (X.shape[0], X.shape[1], 1)).transpose((1, 2, 0))
# X = X.transpose((1, 2, 0))
print(X.shape)

