###############################################################################
# General Information
###############################################################################
# main.py: From here, launch symbolic regression tasks. All
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
S = ['*', '+', '-', '/', 'cos', 'sin']
def main():
    # Load training and test data
    X_constants, X_rnn, y_constants, y_rnn = get_data()

    # Perform the regression task
    results = train(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        operator_list = S,
        # operator_list = ['*','+','-','/','sin','cos','var_x1'],
        min_length = 2,
        max_length = 40,
        type = 'lstm',
        num_layers = 4,
        hidden_size = 250,
        dropout = 0.0,
        lr = 0.0005,
        optimizer = 'lbfgs',
        inner_optimizer = 'lbfgs',
        inner_lr = 0.1,
        inner_num_epochs = 10,
        entropy_coefficient = 0.005,
        risk_factor = 0.95,
        initial_batch_size = 1000,
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



##############################################################################
# Getting Data
##############################################################################

def get_data():
    """Constructs data for model (currently x^3 + x^2 + x)
    """
    # X = np.arange(0.0001, 1.1, 0.01) * 1
    X = np.random.randn(20,1) * 1
    # X = (np.random.rand(20) * 2 - 1) * 2
    # X.sort()
    x1 = X[:,0]
    y = x1**4 + x1**3 + x1**2 + x1

    for j in range(X.shape[1]):
        S.append('var_x'+str(j+1))
    print(S)

    # Split randomly
    comb = list(zip(X, y))
    random.shuffle(comb)
    X, y = zip(*comb)

    X_constants, X_rnn = X,X
    y_constants, y_rnn = y,y
    X_constants, X_rnn = torch.Tensor(X_constants), torch.Tensor(X_rnn)
    y_constants, y_rnn = torch.Tensor(y_constants), torch.Tensor(y_rnn)
    return X_constants, X_rnn, y_constants, y_rnn

if __name__=='__main__':
    main()

