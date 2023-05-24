#数据处理部分之前的代码，加入部分数据处理的库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import gzip
import json
import random
import numpy as np
import pandas as pd

def data_preprocess(data_dir):
    files = os.listdir(data_dir)    #用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    # 第一步，完成数据格式统一
    for f in files:
        # 获取文件路径
        data_file = os.path.join(data_dir)#把目录和文件名合成一个路径
        # 获取文件名后缀
        data_type = os.path.splitext(data_file)[-1]     #分割路径，返回路径名和文件扩展名的元组
        # 获取文件名前缀
        data_name = os.path.splitext(data_file)[0]
        # 如果是excel文件，进行转换
        if data_type == '.xlsx':
            # 需要特别注意的是，在读取excel文件时要指定空值的显示方式，否则会在保存时以字符“.”代替，影响后续的数据分析
            data_xls = pd.read_excel(data_file, index_col=0, na_values='')
            data_xls.to_csv(data_name + '.csv', encoding='utf-8')
            # 顺便删除原文件
            os.remove(data_file)
    # 第二步，完成多文件的合并，文件目录要重新更新一次
    files = os.listdir(data_dir)
    for f in files:
        # 获取文件路径
        data_file = os.path.join(data_dir)
        # 获取文件名前缀
        data_basename = os.path.basename(data_file)
        # 检查风机数据是否有多个数据文件
        if len(data_basename.split('-')) > 1:
            merge_list = []
            # 找出该风机的所有数据文件
            matches = [ f for f in files if (f.find(data_basename.split('-')[0] + '-') > -1)]
            for i in matches:
                # 读取风机这部分数据
                data_df = pd.read_csv(os.path.join(data_dir, i), index_col=False, keep_default_na=False)
                merge_list.append(data_df)
            if len(merge_list) > 0:
                all_data = pd.concat(merge_list,axis=0,ignore_index=True).fillna(".")
                all_data.to_csv(os.path.join(data_dir, data_basename.split('-')[0]+ '.csv'),index=False)
            for i in matches:
                # 删除这部分数据文件
                os.remove(os.path.join(data_dir, i))
            # 更新文件目录
            files = os.listdir(data_dir)
path="D:\code\ContestProject\功率预测竞赛赛题与数据集"
data_preprocess(path)
