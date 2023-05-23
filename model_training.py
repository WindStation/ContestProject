import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime
import paddle
from paddle import nn
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

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # lstm层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            time_major=False)
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = paddle.randn([self.num_layers, x.size(0), self.hidden_size])
        c_0 = paddle.randn([self.num_layers, x.size(0), self.hidden_size])
        output, _ = self.lstm(x, (h_0, c_0))
        pred = self.fc(output)
        pred = pred[:, -1, :]
        return pred
