import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime
import paddle
from paddle import nn
from paddle.nn import functional as F
import paddlets
from paddlets import TSDataset
from paddlets import TimeSeries
from paddlets.models.forecasting import MLPRegressor, LSTNetRegressor
from paddlets.transform import Fill, StandardScaler
from paddlets.metrics import MSE, MAE
from paddlets.analysis import AnalysisReport, Summary
from paddlets.datasets.repository import get_dataset


# 搭建模型
class Net(nn.Layer):

    def __init__(self, input_size, hidden_size, num_layers, output_size, input_len, pred_len, dropout_rate=0.3):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # lstm层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            time_major=False, direction='forward')
        # 全连接层+dropout层
        self.fc_1 = nn.Linear(in_features=input_len * hidden_size, out_features=hidden_size * 2)
        self.fc_2 = nn.Linear(in_features=hidden_size * 2, out_features=pred_len)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # h_0 = paddle.randn([self.num_layers, x.size(0), self.hidden_size])
        # c_0 = paddle.randn([self.num_layers, x.size(0), self.hidden_size])
        # output, _ = self.lstm(x, (h_0, c_0))
        # pred = self.fc(output)
        # pred = pred[:, -1, :]
        # return pred
        output, (hidden, cell) = self.lstm(x)
        output = paddle.reshape(output, [len(output), -1])
        output = self.fc_1(output)
        output = self.dropout(output)
        output = self.fc_2(output)

        return output



#用来按照官方给的测试方法来评价
def calc_acc(y_true, y_pred):
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return 1 - rmse / 201000

# 配置训练参数
input_len = 120 * 4  # 输入序列长度
pred_len = 24 * 4  # 预测序列的长度
input_size = 6  # 输入特征维数为6维，这个是确定的
hidden_size = 12  # 这个参数还有待确定
epoch_num = 100  # 模型训练轮次数
batch_size = 512  # 训练一批次的样本数
loss_rate = 0.001
split_ratio = 0.8

model = Net(input_size, hidden_size, num_layers=1, output_size=2, input_len=input_len, pred_len=pred_len)
criteria = nn.MSELoss()
optimizer = paddle.optimizer.Adam(parameters=model.parameters())
optimizer.set_lr(loss_rate)

# print(model)

# 训练
