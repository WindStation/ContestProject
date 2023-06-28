import paddle
import pandas as pd
import os

import _env.model_training as model_training
from _env.data_preprocess import data_preprocess
from _env.data_preprocess import feature_engineer
from _env.data_loader import TSPredDataset
from _env.utils import from_unix_time


# 有很多的模块因为还未实现  所以有很多未导入
# 对于后期可能需要修改的地方用！！标记

def forecast(df, turbine_id, out_file):
    # 数据预处理  （属于数据读取那边的模块）但有点问题  ！！
    df = data_preprocess(df)
    # 特征工程
    df = feature_engineer(df)
    # 准备数据加载器
    input_len = 120 * 4  # 这里的输入长度 不确定
    pred_len = 24 * 4
    pred_dataset = TSPredDataset(df, input_len=input_len, pred_len=pred_len)
    pred_loader = paddle.io.DataLoader(pred_dataset, shuffle=False, batch_size=1, drop_last=False)
    # 定义模型 (修改为 Net)
    # TODO 下面的参数可能还需要进行调整
    model = model_training.Net(input_len=input_len, input_size=10, hidden_size=48, num_layers=2, pred_len=pred_len,
                               output_size=2)
    # 导入模型权重文件  （这个地方的路径会根据训练的模型权重文件路径改变 ）  ！！
    model.set_state_dict(paddle.load(f'model/model_checkpoint_windid_{turbine_id}.pdparams'))
    model.eval()  # 开启预测
    for batch_id, data in enumerate(pred_loader()):
        x = data[0]
        y = data[1]
        outputs = model(x)
        apower = [x for x in outputs[0].numpy().squeeze()]
        yd15 = [x for x in outputs[1].numpy().squeeze()]
        # 参考代码中 这里的函数是 早停模块下面的 时间戳转换  ！！
        ts_x = [from_unix_time(x) for x in data[2].numpy().squeeze(0)]
        ts_y = [from_unix_time(x) for x in data[3].numpy().squeeze(0)]

    # 这里修改过下一条语句
    result = pd.DataFrame({'DATATIME': ts_y, 'ROUND(A.POWER,0)': apower, 'YD15': yd15})
    # result = pd.DataFrame({'DATATIME': ts_y, 'ROUND(A.POWER,0)': outputs[0].squeeze(), 'YD15': outputs[1].squeeze()})
    result['TurbID'] = turbine_id
    result = result[['TurbID', 'DATATIME', 'ROUND(A.POWER,0)', 'YD15']]
    result.to_csv(out_file, index=False)


if __name__ == "__main__":
    files = os.listdir('../infile')
    # 如果没有这个文件 会创建这个文件 用来存放最终的预测结果
    if not os.path.exists('../pred'):
        os.mkdir('../pred')
    # 第一步，完成数据格式统一
    for f in files:
        if '.csv' not in f:  # TODO 这个地方如果文件格式不是 .csv 就直接跳过了 ？？
            continue

        print(f)
        # 获取文件路径 （获得的是完整路径名）
        data_file = os.path.join('../infile', f)
        print(data_file)
        out_file = os.path.join('../pred', f[:4] + 'out.csv')
        df = pd.read_csv(data_file,
                         parse_dates=['DATATIME'],
                         infer_datetime_format=True,
                         dayfirst=True)
        turbine_id = df.TurbID[0]
        # 预测结果
        forecast(df, turbine_id, out_file)
