# encoding utf-8
'''
@Author: william
@Description: CSV转NPZ
@time:2020/6/16 19:51
'''
import pandas as pd
import numpy as np

data = pd.read_csv('./shuffle_id_list.csv', header=None)
np.save("node_class", data)