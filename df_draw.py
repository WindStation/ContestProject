import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from submission._env.data_preprocess import data_preprocess


def draw(df, turbine_id):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(turbine_id)


def testRead():
    df = pd.read_csv('区域赛数据集/19.csv')
    df = data_preprocess(df)
    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df.dtypes)
    print(df.describe())
    print(df.info())
    print(df.isnull().sum())
    print(df.isnull().sum() / df.shape[0])


if __name__ == '__main__':
    testRead()
