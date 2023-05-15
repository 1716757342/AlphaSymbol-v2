import time
import random
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from operators import Operators
from rnn import DSRRNN
from expression_utils import *
from collections import Counter
from prior import make_prior
from utils import load_config, benchmark, description_length_complexity
import sys
import sympy as sp
import math
import copy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy import*
import csv
# with open("GMST_response_1851-2021.csv", 'r') as x:
#     sample_data = list(csv.reader(x, delimiter=","))
# sample_data = np.array(sample_data)
# data = sample_data[1::]
# index_GLOBAL = data[:, 0] == 'GLOBAL'
# data_GLOBAL = data[index_GLOBAL]
# print(np.shape(data_GLOBAL))
# index_3_GHG = data_GLOBAL[:, 2] == '3-GHG'
# data_3_GHG = data_GLOBAL[index_3_GHG]
# print(np.shape(data_3_GHG))
# index_Total = data_3_GHG[:, 3] == 'Total'
# data_Total = data_3_GHG[index_Total]
# print(np.shape(data_Total))
#
# with open("EMISSIONS_CUMULATIVE_CO2e100_1851-2021.csv", 'r') as x:
#     sample_data_x = list(csv.reader(x, delimiter=","))
# sample_data_x = np.array(sample_data_x)
# data_x = sample_data_x[1::]
# index_GLOBAL_x = data_x[:, 0] == 'GLOBAL'
# data_GLOBAL_x = data_x[index_GLOBAL_x]
# #### CH4
# index_CH4 = data_GLOBAL_x[:, 2] == 'CH[4]'
# data_CH4 = data_GLOBAL_x[index_CH4]
# index_CH4_Total = data_CH4[:, 3] == 'Total'
# data_CH4_Total = data_CH4[index_CH4_Total]
# print(np.shape(data_CH4_Total))
#
# #### CO2
# index_CO2 = data_GLOBAL_x[:, 2] == 'CO[2]'
# data_CO2 = data_GLOBAL_x[index_CO2]
# index_CO2_Total = data_CO2[:, 3] == 'Total'
# data_CO2_Total = data_CO2[index_CO2_Total]
# print(np.shape(data_CO2_Total))
#
# #### N2O
# index_N2O = data_GLOBAL_x[:, 2] == 'N[2]*O'
# data_N2O = data_GLOBAL_x[index_N2O]
# index_N2O_Total = data_N2O[:, 3] == 'Total'
# data_N2O_Total = data_N2O[index_N2O_Total]
#
# print(np.shape(data_N2O_Total))
#
# y = []
# x1 = []
# x2 = []
# x3 = []
# for i in range(2021 - 1950 + 1):
#     # print(i)
#     index_year = data_Total[:, 4] == str(1950 + int(i))
#     index_CH4_year = data_CH4_Total[:, 4] == str(1950 + int(i))
#     index_CO2_year = data_CO2_Total[:, 4] == str(1950 + int(i))
#     index_N2O_year = data_N2O_Total[:, 4] == str(1950 + int(i))
#
#     y.append(np.sum(data_Total[index_year][:, -2].astype('float32')))
#     x1.append(np.sum(data_CH4_Total[index_CH4_year][:, -2].astype('float32')))
#     x2.append(np.sum(data_CO2_Total[index_CO2_year][:, -2].astype('float32')))
#     x3.append(np.sum(data_N2O_Total[index_N2O_year][:, -2].astype('float32')))
# y = np.array(y)
# x1 = np.array(x1).reshape(-1, 1)
# x2 = np.array(x2).reshape(-1, 1)
# x3 = np.array(x3).reshape(-1, 1)
# # print(x1)
# # print(x2)
# # print(x3)
#
#
# # y_pre = 0.0004503275489720375 * (x1+x2+x3) - 7.61169654674424191e-8
# y_pre = 0.000450 * (x1+x2+x3) - 7.61e-8
# plt.plot(y,linestyle ='-',label = 'GMST_test',linewidth=3,markersize = 16,alpha = 0.5)
# plt.plot(y[0:40],linestyle ='-',label = 'GMST_train',linewidth=3,markersize = 16)
# plt.plot(y_pre,c = 'r',linestyle ='--',label = 'GMST_pre',linewidth=3,markersize = 16,alpha = 0.6,dashes=(5, 5))
#
# plt.legend()
# plt.show()

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# print(x1.reshape(1, -1))
# # 创建数据框
# data = {
#     'CH4': list(x1.reshape(1, -1)[0]),
#     'CO2': list(x2.reshape(1, -1)[0]),
#     'N2O': list(x3.reshape(1, -1)[0]),
#     'GMST': list(y.reshape(1, -1)[0])
# }
# df = pd.DataFrame(data)
#
# # 计算相关系数矩阵
# corr_matrix = df.corr()
#
# # 可视化相关系数矩阵
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix of Four Variables')
# plt.show()

a = np.array([np.array([1,2,3,4,5]),np.array([1,2,3,4,5])])
b = np.array([np.array([1,2,3,4,5]),np.array([1,2,3,4,5])])

print(np.sum((a-b)**2))
