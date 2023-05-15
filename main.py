###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip

# Original Paper: https://arxiv.org/abs/1912.04871 (Petersen et al)

# main.py: From here, launch deep symbolic regression tasks. All
# hyperparameters are exposed (info on them can be found in train.py). Unless
# you'd like to impose new constraints / make significant modifications,
# modifying this file (and specifically the get_data function) is likely all
# you need to do for a new symbolic regression task.

###############################################################################
# Dependencies
###############################################################################

from train import train
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from numpy import *
import csv
###############################################################################
# Main Function
###############################################################################

# A note on operators:
# Available operators are: '*', '+', '-', '/', '^', 'sin', 'cos', 'tan',
#   'sqrt', 'square', and 'c.' You may also include constant floats, but they
#   must be strings. For variable operators, you must use the prefix var_.
#   Variable should be passed in the order that they appear in your data, i.e.
#   if your input data is structued [[x1, y1] ... [[xn, yn]] with outputs
#   [z1 ... zn], then var_x should precede var_y.

def main():
    # Load training and test data
    X_constants, X_rnn, y_constants, y_rnn = get_data()

    # Perform the regression task
    results = train(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        operator_list = ['*','+','-','/','sin','cos','exp','ln','sqrt','var_x1','var_x2','var_x3','var_x4','var_x5','c'],
        # operator_list=['*', '+', 'sin', 'var_x'],
        # operator_list=['*', '+', '-', '/', 'cos', 'sin', 'exp', 'ln', 'sqrt', 'var_x1','c'],

        min_length = 2,
        max_length = 40,
        type = 'lstm',
        num_layers = 2,
        hidden_size = 250,
        dropout = 0.0,
        lr = 0.0005,
        optimizer = 'adam',
        inner_optimizer = 'lbfgs',
        inner_lr = 0.1,
        inner_num_epochs = 10,
        entropy_coefficient = 0.005,
        risk_factor = 0.95,
        initial_batch_size = 1000,
        scale_initial_risk = True,
        batch_size = 1000,
        num_batches = 10000,
        use_gpu = False,
        live_print = True,
        summary_print = True,
        config_prior='./config_prior.json'
    )

    # Unpack results
    epoch_best_rewards = results[0]
    epoch_best_expressions = results[1]
    best_reward = results[2]
    best_expression = results[3]
    # Plot best rewards each epoch
    plt.plot([i+1 for i in range(len(epoch_best_rewards))], epoch_best_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Reward over Time')
    plt.show()



###############################################################################
# Getting Data
###############################################################################

# def get_data():
#     """Constructs data for model (currently x^3 + x^2 + x)
#     """
#     X = np.arange(0.0001, 1.1, 0.01) * 20
#     # X = np.random.randn(20) * 2
#     # X = (np.random.rand(20) * 2 - 1) * 2
#     # X.sort()
#     # y = X**7 + X**6 + X**5 + X**4 + X**3 + X**2 + X
#     # y = X ** 6 + X ** 5 + X ** 4 + X ** 3 + X ** 2 + X
#     # y = X**6 + X**5 + X**4 + X ** 3 + X ** 2 + X
#     # y = X ** 4 + X ** 3 + X**2 + X
#     # y = np.sin(X**2) * X**3
#     # y = np.sin(X**2)*np.cos(X) - 1
#     # y = 2.5 * X + 2.7*X**2
#     # y = np.sin(X**2) + np.sin(X**2 + X)
#     # y = 2.4 * X**2 +
#     # plt.plot(X,y)
#     # plt.show()
#     # y = np.log(X ** 2 + 1) + np.log(X+1)
#     # y = np.sin(X**2) * X**3
#
#     x1 = X
#     # y = 0.3*X*sin(2*pi*x1)
#     # y = pow(x1,3)*exp(-x1)*cos(x1)*sin(x1)*(pow(sin(x1),2)*cos(x1)-1)
#     # y = (x1*(x1+1)/2)
#     # y = log(x1 + sqrt(pow(x1, 2) + 1))
#     # y = 0.13*sin(x1)-2.3
#     # y = 3+2.13*log(abs(x1))
#     # y = 3.39*pow(x1,3)+2.12*pow(x1,2)+1.78*x1
#     # y = sin(pow(x1,1))*cos(x1) - 0.8
#     # y = exp(x1)-exp(-1*x1)/2
#     y = sin(pow(x1,3))*cos(pow(x1,2))-1
#     y = log(x1+1)+log(pow(x1,2)+1)+log(x1)
#     y = sin(pow(x1,2))*cos(x1)-5
#     y = exp(-0.5*pow(x1,2))
#     y = log(x1+1)+log(pow(x1,2)+1) ####本机
#     # y = (pow(x1+1,3))/(pow(x1,2)-x1+1)
#     y = sin(pow(x1,2))*cos(x1)-1 ####RSRex
#     X = X.reshape(X.shape[0], 1)
#
#     # Split randomly
#     comb = list(zip(X, y))
#     random.shuffle(comb)
#     X, y = zip(*comb)
#
#     # Proportion used to train constants versus benchmarking functions
#     # training_proportion = 0.2
#     # div = int(training_proportion*len(X))
#     # X_constants, X_rnn = np.array(X[:div]), np.array(X[div:])
#     # y_constants, y_rnn = np.array(y[:div]), np.array(y[div:])
#     X_constants, X_rnn = X,X
#     y_constants, y_rnn = y,y
#     X_constants, X_rnn = torch.Tensor(X_constants), torch.Tensor(X_rnn)
#     y_constants, y_rnn = torch.Tensor(y_constants), torch.Tensor(y_rnn)
#     return X_constants, X_rnn, y_constants, y_rnn
#
# if __name__=='__main__':
#     main()


###############################################################################
# Getting Data
###############################################################################

def get_data():
    """Constructs data for model (currently x^3 + x^2 + x)
    """
    # X = np.arange(-1, 1.1, 0.1)
    #
    # X1 = np.arange(-1, 1.1, 0.05) * 1
    # X2 = np.arange(-1, 1.1, 0.05) * 1
    # X1 = (np.random.rand(100) * 2 - 1) * 4
    # X2 = (np.random.rand(100) * 2 - 1) * 4
    X1 = np.random.rand(40) * 1
    X2 = np.random.rand(40) * 1
    X3 = np.random.rand(40) * 1
    X4 = np.random.rand(40) * 1
    X5 = np.random.rand(40) * 1
    # X6 = np.random.rand(40) * 1

    # y = (X1*X2)/(4*3.14 * X3 * X4**2)
    # y = 0.5 * X1 * (X2**2 + X3**2 + X4**2)
    # y = (X1 - X2*X3)/(sqrt(1-X2**2/X4))
    # y = X1 * X2 * X3 * (1/X4 - 1/X5)
    y = (X1 * X2)/X4 * (X3**2 - X5**2)
    # y = (4*3.14 * X1 * X2**2)/(X3 * X4**2)
    # y = 1/(X1-1) *(X2*X3*X4)
    # y = sqrt(X1**2 + X2**2 - 2 * X1 * X2*cos(X5-X6))
    # y = (0.5 * X1 * X2 *X3**2) *(8 * 3.14 * X4**2/3) * (X5**4/(X5**2 - X6**2)**2)
    # y = (X1 * X2**3)/(3.14*3.14 * X3**2 *(exp((X1*X2)/(X4*X5))-1))
    # y = X1 * exp(-(X2*X3*X4)/(X5*X6))

    num = len(X1)
    X1 = X1.reshape(num, 1)
    X2 = X2.reshape(num, 1)
    X3 = X3.reshape(num, 1)
    X4 = X4.reshape(num, 1)
    X5 = X5.reshape(num, 1)
    # X6 = X6.reshape(num, 1)

    X = np.concatenate((X1, X2, X3, X4, X5), axis=1)
    comb = list(zip(X, y))

    # Proportion used to train constants versus benchmarking functions
    training_proportion = 0.2
    div = int(training_proportion*len(X))
    # X_constants, X_rnn = np.array(X[:div]), np.array(X[div:])
    # y_constants, y_rnn = np.array(y[:div]), np.array(y[div:])
    X_constants, X_rnn = X, X
    y_constants, y_rnn = y, y
    X_constants, X_rnn = torch.from_numpy(X_constants), torch.Tensor(X_rnn)
    y_constants, y_rnn = torch.from_numpy(y_constants), torch.Tensor(y_rnn)
    return X_constants, X_rnn, y_constants, y_rnn

if __name__=='__main__':
    main()


###############################################################################
# Getting Data
###############################################################################

# def get_data():
#     """Constructs data for model (currently x^3 + x^2 + x)
#     """
#
#
#     with open("GMST_response_1851-2021.csv", 'r') as x:
#         sample_data = list(csv.reader(x, delimiter=","))
#     sample_data = np.array(sample_data)
#     data = sample_data[1::]
#     index_GLOBAL = data[:,0] == 'GLOBAL'
#     data_GLOBAL = data[index_GLOBAL]
#     print(np.shape(data_GLOBAL))
#     index_3_GHG = data_GLOBAL[:,2] == '3-GHG'
#     data_3_GHG = data_GLOBAL[index_3_GHG]
#     print(np.shape(data_3_GHG))
#     index_Total = data_3_GHG[:,3] == 'Total'
#     data_Total = data_3_GHG[index_Total]
#     print(np.shape(data_Total))
#
#     with open("EMISSIONS_CUMULATIVE_CO2e100_1851-2021.csv", 'r') as x:
#         sample_data_x = list(csv.reader(x, delimiter=","))
#     sample_data_x = np.array(sample_data_x)
#     data_x = sample_data_x[1::]
#     index_GLOBAL_x= data_x[:, 0] == 'GLOBAL'
#     data_GLOBAL_x = data_x[index_GLOBAL_x]
#     #### CH4
#     index_CH4 = data_GLOBAL_x[:,2] == 'CH[4]'
#     data_CH4 = data_GLOBAL_x[index_CH4]
#     index_CH4_Total = data_CH4[:,3] == 'Total'
#     data_CH4_Total = data_CH4[index_CH4_Total]
#     print(np.shape(data_CH4_Total))
#
#     #### CO2
#     index_CO2 = data_GLOBAL_x[:,2] == 'CO[2]'
#     data_CO2 = data_GLOBAL_x[index_CO2]
#     index_CO2_Total = data_CO2[:,3] == 'Total'
#     data_CO2_Total = data_CO2[index_CO2_Total]
#     print(np.shape(data_CO2_Total))
#
#     #### N2O
#     index_N2O = data_GLOBAL_x[:,2] == 'N[2]*O'
#     data_N2O = data_GLOBAL_x[index_N2O]
#     index_N2O_Total = data_N2O[:,3] == 'Total'
#     data_N2O_Total = data_N2O[index_N2O_Total]
#
#     print(np.shape(data_N2O_Total))
#
#     y = []
#     x1 = []
#     x2 = []
#     x3 = []
#     for i in range(2021-1950 + 1):
#         # print(i)
#         index_year = data_Total[:,4] == str(1950 + int(i))
#         index_CH4_year = data_CH4_Total[:,4] == str(1950 + int(i))
#         index_CO2_year = data_CO2_Total[:, 4] == str(1950 + int(i))
#         index_N2O_year = data_N2O_Total[:, 4] == str(1950 + int(i))
#
#         y.append(np.sum(data_Total[index_year][:,-2].astype('float32')))
#         x1.append(np.sum(data_CH4_Total[index_CH4_year][:,-2].astype('float32')))
#         x2.append(np.sum(data_CO2_Total[index_CO2_year][:, -2].astype('float32')))
#         x3.append(np.sum(data_N2O_Total[index_N2O_year][:, -2].astype('float32')))
#     y = np.array(y)
#     x1 = np.array(x1).reshape(-1,1)
#     x2 = np.array(x2).reshape(-1,1)
#     x3 = np.array(x3).reshape(-1,1)
#
#     X = np.concatenate((x1, x2, x3), axis=1)
#     print(X)
#     print(y)
#     # Split randomly
#     comb = list(zip(X, y))
#
#     # Proportion used to train constants versus benchmarking functions
#     training_proportion = 0.2
#     div = int(training_proportion * len(X))
#     # X_constants, X_rnn = np.array(X[:div]), np.array(X[div:])
#     # y_constants, y_rnn = np.array(y[:div]), np.array(y[div:])
#     X_constants, X_rnn = X, X
#     y_constants, y_rnn = y, y
#     X_constants, X_rnn = torch.from_numpy(X_constants), torch.Tensor(X_rnn)
#     y_constants, y_rnn = torch.from_numpy(y_constants), torch.Tensor(y_rnn)
#     return X_constants, X_rnn, y_constants, y_rnn
# if __name__=='__main__':
#     main()