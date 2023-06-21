import os

import numpy as np
import paddle
import pandas as pd
from matplotlib import pyplot as plt
from paddle import nn
from tqdm import tqdm

# TODO 可能要修改
from submission._env.data_loader import TSDataset
from submission._env.data_preprocess import data_preprocess, feature_engineer
from submission._env.utils import EarlyStopping, from_unix_time


# 搭建模型
class Net(nn.Layer):

    def __init__(self, input_size, hidden_size, num_layers, output_size, input_len, pred_len, dropout_rate=0.7):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # lstm层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            time_major=False, direction='forward')
        # ROUND字段的全连接层
        self.fc1_1 = nn.Linear(in_features=input_len * hidden_size, out_features=hidden_size * 2)
        self.fc1_2 = nn.Linear(in_features=hidden_size * 2, out_features=pred_len)
        # YD15的全连接层
        self.fc2_1 = nn.Linear(in_features=input_len * hidden_size, out_features=hidden_size * 2)
        self.fc2_2 = nn.Linear(in_features=hidden_size * 2, out_features=pred_len)
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
        # output = self.fc_1(output)
        # output = self.dropout(output)
        # output = self.fc_2(output)
        output1 = self.fc1_1(output)
        output1 = self.dropout(output1)
        output1 = self.fc1_2(output1)

        output2 = self.fc2_1(output)
        output2 = self.dropout(output2)
        output2 = self.fc2_2(output2)

        return [output1, output2]


class Loss(nn.Layer):

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, inputs, labels):
        mse_loss = nn.loss.MSELoss()
        mse_1 = mse_loss(inputs[0], labels[:, :, 0].squeeze(-1))
        mse_2 = mse_loss(inputs[1], labels[:, :, 1].squeeze(-1))
        # 设置权重偏向YD15
        return mse_1, mse_2, 0.3 * mse_1 + 0.7 * mse_2


def calc_acc(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return 1 - rmse / 201000


# 配置训练参数
input_len = 120 * 4  # 输入序列长度
pred_len = 24 * 4  # 预测序列的长度
input_size = 10  # 输入特征维数为6维，这个是确定的
hidden_size = 48  # 这个参数还有待确定
epoch_num = 100  # 模型训练轮次数
batch_size = 512  # 训练一批次的样本数
loss_rate = 0.001
split_ratio = 0.8
learning_rate = 0.001  # 学习率
patience = 15  # 如果连续patience个轮次性能没有提升，就会停止训练


def train(df, turbine_id):
    # 设置数据集
    train_dataset = TSDataset(df, input_len=input_len, pred_len=pred_len, data_type='train')
    val_dataset = TSDataset(df, input_len=input_len, pred_len=pred_len, data_type='val')
    test_dataset = TSDataset(df, input_len=input_len, pred_len=pred_len, data_type='test')
    print(f'LEN | train_dataset:{len(train_dataset)}, val_dataset:{len(val_dataset)}, test_dataset:{len(test_dataset)}')

    # 设置数据读取器
    train_loader = paddle.io.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    val_loader = paddle.io.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    test_loader = paddle.io.DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=False)

    # 设置模型
    model = Net(input_size, hidden_size, num_layers=2, output_size=2, input_len=input_len, pred_len=pred_len)

    # 设置优化器
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=learning_rate, factor=0.5, patience=3, verbose=True)
    opt = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())

    # 设置损失函数，使用自定义的损失处理类，下面对应调用都要更改
    # criteria = nn.MSELoss()
    double_loss = Loss()

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []
    early_stopping = EarlyStopping(patience=patience, turb_id=turbine_id, verbose=True)

    for epoch in tqdm(range(epoch_num)):
        # =====================train============================
        # train_epoch_loss, train_epoch_mse1, train_epoch_mse2 = [], [], []  直接使用
        train_epoch_loss, train_epoch_mse1, train_epoch_mse2 = [], [], []
        model.train()  # 开启训练
        for batch_id, data in enumerate(train_loader()):
            x = data[0]
            y = data[1]
            # print(f'x:\n{x}\ny:\n{y}')
            # 预测
            outputs = model(x)
            # print(outputs)
            # 计算损失
            mse_1, mse_2, avg_loss = double_loss(outputs, y)
            # 反向传播
            avg_loss.backward()
            # 梯度下降
            opt.step()
            # 清空梯度
            opt.clear_grad()
            train_epoch_loss.append(avg_loss.numpy()[0])
            train_loss.append(avg_loss.item())
            train_epoch_mse1.append(mse_1.item())
            train_epoch_mse2.append(mse_2.item())

        train_epochs_loss.append(np.average(train_epoch_loss))
        print("epoch={}/{} of train | loss={}".format(epoch, epoch_num, np.average(train_epoch_loss)))

        # =====================valid============================
        model.eval()  # 开启评估/预测
        valid_epoch_loss, valid_epochs_mse1, valid_epochs_mse2 = [], [], []
        for batch_id, data in enumerate(val_loader()):
            x = data[0]
            y = data[1]
            outputs = model(x)
            mse_1, mse_2, avg_loss = double_loss(outputs, y)
            valid_epoch_loss.append(avg_loss.numpy()[0])
            valid_loss.append(avg_loss.numpy()[0])
            valid_epochs_mse1.append(mse_1.item())
            valid_epochs_mse2.append(mse_2.item())

        valid_epochs_loss.append(np.average(valid_epoch_loss))
        print('Valid:MSE of YD15:{}'.format(np.average(train_epoch_loss)))

        # ==================early stopping======================
        early_stopping(valid_epochs_loss[-1], model=model)
        if early_stopping.early_stop:
            print(f"Early stopping at Epoch {epoch - patience}")
            break

    print('Train & Valid: ')
    plt.figure(figsize=(12, 3))
    plt.subplot(121)
    plt.plot(train_loss[:], label="train")
    plt.title("train_loss")
    plt.xlabel('iteration')
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:], '-o', label="train")
    plt.plot(valid_epochs_loss[1:], '-o', label="valid")
    plt.title("epochs_loss")
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =====================test============================
    # 加载最优epoch节点下的模型
    model = Net(input_size, hidden_size, num_layers=2, output_size=2, input_len=input_len, pred_len=pred_len)
    model.set_state_dict(paddle.load(f'../model/model_checkpoint_windid_{turbine_id}.pdparams'))

    model.eval()  # 开启评估/预测
    test_loss, test_epoch_mse1, test_epoch_mse2 = [], [], []
    test_accs1, test_accs2 = [], []
    for batch_id, data in tqdm(enumerate(test_loader())):
        x = data[0]
        y = data[1]
        ts_y = [from_unix_time(x) for x in data[3].numpy().squeeze(0)]
        outputs = model(x)
        mse_1, mse_2, avg_loss = double_loss(outputs, y)
        acc1 = calc_acc(y.numpy().squeeze(0)[:, 0], outputs[0].numpy().squeeze(0))
        acc2 = calc_acc(y.numpy().squeeze(0)[:, 1], outputs[1].numpy().squeeze(0))
        test_loss.append(avg_loss.numpy()[0])
        test_epoch_mse1.append(mse_1.numpy()[0])
        test_epoch_mse2.append(mse_2.numpy()[0])
        test_accs1.append(acc1)
        test_accs2.append(acc2)
    print('Test: ')
    print('MSE of ROUND(A.POWER,0):{}, MSE of YD15:{}'.format(np.average(test_epoch_mse1), np.average(test_epoch_mse2)))
    print('Mean MSE:', np.mean(test_loss))
    print('ACC of ROUND(A.POWER,0):{}, ACC of YD15:{}'.format(np.average(test_accs1), np.average(test_accs2)))


# 下面提交的时候注释掉
# data_path = 'E:\竞赛\软件杯\ContestProject\功率预测竞赛赛题与数据集'
if __name__ == '__main__':
    paddle.device.set_device('gpu:0')
    data_path = '../../功率预测竞赛赛题与数据集'
    files = os.listdir(data_path)
    debug = False  # 为了快速跑通代码，可以先尝试用采样数据做debug

    # 遍历每个风机的数据做训练、验证和测试
    for f in files:
        df = pd.read_csv(os.path.join(data_path, f),
                         parse_dates=['DATATIME'],
                         infer_datetime_format=True,
                         dayfirst=True)
        turbine_id = int(float(f.split('.csv')[0]))
        print(f'turbine_id:{turbine_id}')

        if debug:
            df = df.iloc[-24 * 4 * 200:, :]

        # 数据预处理
        df = data_preprocess(df)
        # 特征工程
        df = feature_engineer(df)

        # 训练模型
        train(df, turbine_id)

