import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset

import warnings

warnings.filterwarnings('ignore')


def data_preprocess(df):
    """数据预处理：
        1、读取数据
        2、数据排序
        3、去除重复值
        4、重采样（可选）
        5、缺失值处理
        6、异常值处理
    """
    # ===========读取数据===========
    df = df.sort_values(by='DATATIME', ascending=True)
    print('df.shape:', df.shape)
    print(f"Time range from {df['DATATIME'].values[0]} to {df['DATATIME'].values[-1]}")

    # ===========去除重复值===========
    df = df.drop_duplicates(subset='DATATIME', keep='first')
    print('After Dropping dulicates:', df.shape)

    # ===========重采样（可选） + 线性插值===========
    df = df.set_index('DATATIME')
    # 重采样（可选）：比如04风机缺少2022-04-10和2022-07-25两天的数据，重采样会把这两天数据补充进来
    # df = df.resample(rule=to_offset('15T').freqstr, label='right', closed='right').interpolate(method='linear', limit_direction='both').reset_index()

    # ===========异常值处理===========
    # 当实际风速为0时，功率置为0
    df.loc[df['ROUND(A.WS,1)'] == 0, 'YD15'] = 0

    # 根据阈值，过大或过小的值视为异常值，去除，变成缺失值，到后面处理
    df.loc[(df['ROUND(A.WS,1)'] > 1e10) | (df['ROUND(A.WS,1)'] < -1e10), 'ROUND(A.WS,1)'] = None
    df.loc[(df['ROUND(A.POWER,0)'] > 1e10) | (df['ROUND(A.POWER,0)'] < -1e10), 'ROUND(A.POWER,0)'] = None

    # ===========缺失值处理===========
    for idx, data in df.iterrows():
        if data['ROUND(A.WS,1)'] is None:
            data['ROUND(A.WS,1)'] = data['WINDSPEED']

        if data['YD15'] is None:
            if data['ROUND(A.POWER,0)'] is not None:
                data['YD15'] = data['ROUND(A.POWER,0)']
            else:
                data['YD15'] = data['PREPOWER']

        if data['ROUND(A.POWER,0)'] is None:
            if data['YD15'] is not None:
                data['ROUND(A.POWER,0)'] = data['YD15']
            else:
                data['ROUND(A.POWER,0)'] = data['PREPOWER']

    # 可选：尝试一些其他缺失值处理方式，比如，用同时刻附近风机的值求均值填补缺失值
    df = df.interpolate(method='linear', limit_direction='both').reset_index()
    print('After Resampling:', df.shape)


    # 可选： 风速过大但功率为0的异常：先设计函数拟合出：实际功率=f(风速)，
    # 然后代入异常功率的风速获取理想功率，替换原异常功率

    # 可选： 对于在特定风速下的离群功率（同时刻用IQR检测出来），做功率修正（如均值修正）
    return df


def feature_engineer(df):
    """特征工程：时间戳特征"""
    # 时间戳特征
    df['month'] = df.DATATIME.apply(lambda row: row.month, 1)
    df['day'] = df.DATATIME.apply(lambda row: row.day, 1)
    df['weekday'] = df.DATATIME.apply(lambda row: row.weekday(), 1)
    df['hour'] = df.DATATIME.apply(lambda row: row.hour, 1)
    df['minute'] = df.DATATIME.apply(lambda row: row.minute, 1)

    # 可选： 挖掘更多特征：差分序列、同时刻风场/邻近风机的特征均值/标准差等
    return df
