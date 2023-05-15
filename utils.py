import importlib
import json
import os
import random
import re
import time

import numpy as np
import sympy
import omegaconf
import torch
from sympy import parse_expr, preorder_traversal, count_ops, lambdify, N
from torch import nn
# from utils import load_config, benchmark, description_length_complexity
# from prior import make_prior


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def generateDataFast(eq, n_points, n_vars, decimals, min_x, max_x,
                     total_variabels=['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9']):
    '''
    逐点计算
    :param eq: expression str with consts
    :param n_points: number of points
    :param min_x: min x
    :param max_x: max x
    :return:
    '''
    X_ls = []
    Y_ls = []
    # X_dict = {}
    # temp = []
    start_time = time.time()
    p = 0
    while p < n_points:
        X_dict = {}
        temp = []
        # 采样一组点
        X = np.round(np.random.uniform(min_x, max_x, n_vars), decimals=decimals)

        # 将点与自变量进行map
        for x in total_variabels:
            if x in eq:
                X_dict.update({x: X[eval(x[-1]) - 1]})
                temp.append(x)
            else:
                # X[-1] = 0.00000000
                pass

        try:
            y = sympy.lambdify(",".join(temp), eq)(**X_dict)
        except:
            return [], []

        # 若超过10秒还算不出正确的y值，说明表达式大概率无意义，直接返回空列表，重新生成新的表达式
        time_cost = time.time() - start_time
        if time_cost > 2.5:
            return [], []

        # 若有NaN或inf就重新计算
        if np.isnan(y) or np.isinf(y):
            # print('y值计算错误，重新采样计算...')
            continue

        y = float(np.round(y, decimals=decimals))  # 保留小数点后数位并保证是浮点数
        y = y if abs(y) < 5e4 else np.sign(y) * 5e4  # 将最大的y值限制在阈值范围内
        y = abs(y) if np.iscomplex(y) else y

        p += 1

        X_ls.append(list(X))
        Y_ls.append(y)

    # print(X_ls, '\n', shape(X_ls))
    # print(Y_ls, '\n', shape(Y_ls))
    return X_ls, Y_ls


def import_custom_source(import_source):
    """
    Provides a way to import custom modules. The return will be a reference to the desired source
    Parameters
    ----------
        import_source : import path
            Source to import from, for most purposes: <module_name>:<class or function name>

    Returns
    -------
        mod : ref
            reference to the imported module
    """

    # Partially validates if the import_source is in correct format
    regex = '[\w._]+:[\w._]+'  # lib_name:class_name
    m = re.match(pattern=regex, string=import_source)
    # Partial matches mean that the import will fail
    assert m is not None and m.end() == len(
        import_source), "*** Failed to import malformed source string: " + import_source

    source, type = import_source.split(':')

    # Dynamically imports the configured source
    mod = importlib.import_module(source)
    func = getattr(mod, type)

    return func


def load_config(config_path):
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    return config


###############################################################################
# Reward function
###############################################################################

def benchmark(expression, X_rnn, y_rnn, reward_type=None, cur_epochs=None):
    """Obtain reward for a given expression using the passed X_rnn and y_rnn
    """
    if reward_type == "MSE":
        with torch.no_grad():
            y_pred = expression(X_rnn)
            return reward_mse(y_pred, y_rnn)

    if reward_type == "NRMSE":
        with torch.no_grad():
            y_pred = expression(X_rnn)
            return reward_nrmse(y_pred, y_rnn)

    if reward_type == "MEDL":
        reward_medl = mean_error_description_length(expression, X_rnn, y_rnn)
        return reward_medl  # medl越小越好，reward要求要越来越大，所以取负

    if reward_type == "Pareto_optimal":
        # medl = mean_error_description_length(str(expression), X_rnn, y_rnn)
        # complexity = description_length_complexity(str(expression))
        # # 近似帕累托最优 -> 寻找Pareto stationary point -> 最小化各目标函数的梯度加权和的二范数，s.t.权重和为1 -> 取负号作为reward使其最大化
        # w_1 = 0.9
        # w_2 = 0.1
        # reward = -(w_1 * medl + w_2 * complexity)
        # print("Weighted MEDL: ", w_1*medl, "\n", "Weighted Complexity: ", w_2*complexity)

        with torch.no_grad():
            y_pred = expression(X_rnn)
            loss = nn.MSELoss()
            val = torch.sqrt(loss(y_pred, y_rnn))  # Convert to RMSE
            val = torch.std(y_rnn) * val  # Normalize using stdev of targets
            nrmse = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip

        complexity = description_length_complexity(str(expression))
        # 动态调整
        w_1 = 1.0
        w_2 = 1e-5
        reward = 1 / (1 + (w_1 * nrmse + w_2 * 10**(cur_epochs//10) * complexity))  # 每10个epoch, w_2增加10倍
        # print("nrmse: ", nrmse)
        # print("DL: ", complexity)
        if cur_epochs is not None and cur_epochs % 10 == 0:
            print("epoch: {}, w_2: {}".format(cur_epochs, w_2 * 10**(cur_epochs//10)))

        # w_1 = 1.0
        # w_2 = 0.01
        # reward = 1 / (1 + (w_1 * nrmse + w_2 * complexity))  # 每10个epoch, w_2增加10倍
        # if cur_epochs is not None and cur_epochs % 10 == 0:
        #     print("epoch: {}, w_2: {}".format(cur_epochs, w_2))
        # print("nrmse: ", nrmse)
        # print("DL: ", complexity)

        return reward

    if reward_type == "R^2":
        with torch.no_grad():
            y_pred = expression(X_rnn)
        return R_Square(y_pred, y_rnn)


def R_Square(y_pred, y_rnn):
    return (1 - torch.sum(torch.square(y_pred - y_rnn)) / torch.sum(torch.square(y_rnn - torch.mean(y_rnn)))).item()


def reward_mse(y_pred, y_rnn):
    loss = nn.MSELoss()(y_pred, y_rnn)
    val = min(torch.nan_to_num(loss, nan=1e10), torch.tensor(1e10))
    val = 1 / (1 + val)
    return val.item()


def reward_nrmse(y_pred, y_rnn):
    """Compute NRMSE between predicted y and actual y
    """
    loss = nn.MSELoss()
    val = torch.sqrt(loss(y_pred, y_rnn))  # Convert to RMSE
    val = torch.std(y_rnn) * val  # Normalize using stdev of targets
    val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip
    # print("NRMSE: ", val)
    val = 1 / (1 + val)  # Squash
    return val.item()


def mean_error_description_length(expression, X, y):
    """mean error description length is [0, inf]"""
    try:
        with torch.no_grad():
            y_pred = expression(X).numpy()
            y = y.numpy()
            # Remove accidental nan's
            good_idx = np.where(np.isnan(y_pred) == False)

            # use this to get rid of cases where the loss gets complex because of transformations of the output variable
            if isinstance(np.mean((y_pred - y) ** 2), complex):
                return 1000000
            else:
                # val = min(np.nan_to_num(abs(y_pred - y), nan=1e5), np.array(1e10))
                error = np.mean(np.log2(1 + abs(y_pred[good_idx] - y[good_idx])))
                mdle = min(np.nan_to_num(error, nan=1e6), np.array(1e10))  # Fix nan and clip
                # print("Mean Description Length Error: ", mdle)
                val = 1 / (1 + mdle)
                return val.item()

    except:
        return 1000000


def description_length_complexity(eqn):
    """计算表达式的DL复杂度，包括常数和符号"""
    expr = parse_expr(eqn)
    is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
    numbers_expr = [subexpression for subexpression in preorder_traversal(expr) if
                    is_atomic_number(subexpression)]
    complity = 0
    for j in numbers_expr:
        try:
            complity = complity + get_number_DL_snapped(float(j))  # 计算表达式中常数的DL复杂度
        except:
            complity = complity + 1000000

    # 加上符号的复杂度
    n_variables = len(expr.free_symbols)
    n_operations = len(count_ops(expr, visual=True).free_symbols)
    if n_operations != 0 or n_variables != 0:
        complity = complity + (n_variables + n_operations) * np.log2((n_variables + n_operations))
    return complity


def get_number_DL_snapped(n):
    epsilon = 1e-10
    n = float(n)
    if np.isnan(n):
        return 1000000
    elif np.abs(n - int(n)) < epsilon:
        return np.log2(1 + abs(int(n)))
    elif np.abs(n - bestApproximation(n, 10000)[0]) < epsilon:
        _, numerator, denominator, _ = bestApproximation(n, 10000)
        return np.log2((1 + abs(numerator)) * abs(denominator))
    elif np.abs(n - np.pi) < epsilon:
        return np.log2(1 + 3)
    else:
        PrecisionFloorLoss = 1e-14
        return np.log2(1 + (float(n) / PrecisionFloorLoss) ** 2) / 2


def bestApproximation(x, imax):
    # The input is a numpy parameter vector p.
    # The output is an integer specifying which parameter to change,
    # and a float specifying the new value.
    def float2contfrac(x, nmax):
        x = float(x)
        c = [np.floor(x)];
        y = x - np.floor(x)
        k = 0
        while np.abs(y) != 0 and k < nmax:
            y = 1 / float(y)
            i = np.floor(y)
            c.append(i)
            y = y - i
            k = k + 1
        return c

    def contfrac2frac(seq):
        ''' Convert the simple continued fraction in `seq`
            into a fraction, num / den
            '''
        num, den = 1, 0
        for u in reversed(seq):
            num, den = den + num * u, num
        return num, den

    def contFracRationalApproximations(c):
        return np.array(list(contfrac2frac(c[:i + 1]) for i in range(len(c))))

    def contFracApproximations(c):
        q = contFracRationalApproximations(c)
        return q[:, 0] / float(q[:, 1])

    def truncateContFrac(q, imax):
        k = 0
        while k < len(q) and np.maximum(np.abs(q[k, 0]), q[k, 1]) <= imax:
            k = k + 1
        return q[:k]

    def pval(p):
        p = p.astype(float)
        return 1 - np.exp(-p ** 0.87 / 0.36)

    xsign = np.sign(x)
    q = truncateContFrac(contFracRationalApproximations(float2contfrac(abs(x), 20)), imax)

    if len(q) > 0:
        p = np.abs(q[:, 0] / q[:, 1] - abs(x)).astype(float) * (1 + np.abs(q[:, 0])) * q[:, 1]
        p = pval(p)
        i = np.argmin(p)
        return (xsign * q[i, 0] / float(q[i, 1]), xsign * q[i, 0], q[i, 1], p[i])
    else:
        return (None, 0, 0, 1)
