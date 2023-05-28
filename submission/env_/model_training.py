import paddle
from paddle import nn


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


