###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip

# Original Paper: https://arxiv.org/abs/1912.04871 (Petersen et al)

# train.py: Contains main training loop (and reward functions) for PyTorch
# implementation of Deep Symbolic Regression.

###############################################################################
# Dependencies
###############################################################################

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
###############################################################################
# Main Training loop
###############################################################################

def train(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        operator_list = ['*', '+', '-', '/', '^', 'cos', 'sin', 'c', 'var_x'],
        min_length = 2,
        max_length = 12,
        type = 'lstm',
        num_layers = 1,
        dropout = 0.0,
        lr = 0.0005,
        optimizer = 'adam',
        inner_optimizer = 'rmsprop',
        inner_lr = 0.1,
        inner_num_epochs = 15,
        entropy_coefficient = 0.005,
        risk_factor = 0.95,
        initial_batch_size = 500, ##2000
        scale_initial_risk = True,
        batch_size = 500,
        num_batches = 200,
        hidden_size = 500,
        use_gpu = False,
        live_print = True,
        summary_print = True,
        config_prior=None,
    ):
    """Deep Symbolic Regression Training Loop

    ~ Parameters ~
    - X_constants (Tensor): X dataset used for training constants
    - y_constants (Tensor): y dataset used for training constants
    - X_rnn (Tensor): X dataset used for obtaining reward / training RNN
    - y_rnn (Tensor): y dataset used for obtaining reward / training RNN
    - operator_list (list of str): operators to use (all variables must have prefix var_)
    - min_length (int): minimum number of operators to allow in expression
    - max_length (int): maximum number of operators to allow in expression
    - type ('rnn', 'lstm', or 'gru'): type of architecture to use
    - num_layers (int): number of layers in RNN architecture
    - dropout (float): dropout (if any) for RNN architecture
    - lr (float): learning rate for RNN
    - optimizer ('adam' or 'rmsprop'): optimizer for RNN
    - inner_optimizer ('lbfgs', 'adam', or 'rmsprop'): optimizer for expressions
    - inner_lr (float): learning rate for constant optimization
    - inner_num_epochs (int): number of epochs for constant optimization
    - entropy_coefficient (float): entropy coefficient for RNN
    - risk_factor (float, >0, <1): we discard the bottom risk_factor quantile
      when training the RNN
    - batch_size (int): batch size for training the RNN
    - num_batches (int): number of batches (will stop early if found)
    - hidden_size (int): hidden dimension size for RNN
    - use_gpu (bool): whether or not to train with GPU
    - live_print (bool): if true, will print updates during training process

    ~ Returns ~
    A list of four lists:
    [0] epoch_best_rewards (list of float): list of highest reward obtained each epoch
    [1] epoch_best_expressions (list of Expression): list of best expression each epoch
    [2] best_reward (float): best reward obtained
    [3] best_expression (Expression): best expression obtained
    """

    AVAILABLE_CHOICES = operator_list
    # AVAILABLE_CHOICES = ['*', '+', 'sin', 'var_x']
    AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)
    MAX_ROUND_NUMBER = 30

    # ind = 6

    class Node(object):
        def __init__(self):
            self.parent = None
            self.children = []
            self.visit_times = 0
            self.quality_value = 0.0
            self.state = None

        def set_state(self, state):
            self.state = state

        def get_state(self):
            return self.state

        def set_parent(self, parent):
            self.parent = parent

        def get_parent(self):
            return self.parent

        def set_children(self, children):
            self.children = children

        def get_children(self):
            return self.children

        def get_visit_times(self):
            return self.visit_times

        def set_visit_times(self, times):
            self.visit_times = times

        def visit_times_add_one(self):
            self.visit_times += 1

        def get_quality_value(self):
            return self.quality_value

        def set_quality_value(self, value):
            self.quality_value = value

        def quality_value_add_n(self, r):
            self.quality_value += r

        def is_all_expand(self):
            if len(self.children) == AVAILABLE_CHOICE_NUMBER:
                return True
            else:
                return False

        def add_child(self, sub_node):
            sub_node.set_parent(self)
            self.children.append(sub_node)

        def __repr__(self):
            # return "Node:{},Q/N:{}/{},state:{}".format(hash(self),self.quality_value,self.visit_times,self.state.current_value)
            return "Node:{},Q/N:{}/{},statevalue:{},statereward:{}".format(hash(self), self.quality_value,
                                                                           self.visit_times,
                                                                           self.state.current_value,
                                                                           self.state.compute_reward())

    class State(object):  # 某游戏的状态，例如模拟一个数相加等于1的游戏
        def __init__(self):
            self.current_value = 0.0  # 当前数
            self.current_round_index = 0  # 第几轮(长度）
            self.cumulative_choices = []  # 选择过程记录
            self.counter = 1
            # self.ind = 6

        def is_terminal(self):  # 判断游戏是否结束
            if self.counter == 0:
                # if self.counter == 0:
                return True
            else:
                return False

        def compute_reward(self):  # 当前得分，越接近1分值越高
            return reward_nrmse(self.current_value, y_rnn)
            # return 0.99 ** (len(self.cumulative_choices)) / np.sqrt(1 + (np.sum((y_rnn - self.current_value) ** 2)) / len(X_rnn))
            # return -np.sum(abs(y_1 - self.current_value)) / num

        def set_current_value(self, value):
            self.current_value = value

        def set_current_round_index(self, round):
            self.current_round_index = round

        def set_cumulative_choices(self, choices):
            self.cumulative_choices = choices

        def set_counter(self, coun, s):
            self.counter = coun + Arity(s) - 1

        def get_next_state_with_random_choice(self, SS):  # 得到下个状态
            # random_choice = random.choice([choice for choice in AVAILABLE_CHOICES] , p= PP)
            random_choice = SS
            # ind = IND - 1
            # print('random_choice',random_choice)
            # print(random_choice)
            next_state = State()
            next_state.set_counter(self.counter, random_choice)
            next_state.set_cumulative_choices(self.cumulative_choices + [random_choice])  ## 所选值的列表
            # if next_state.counter == 0:
            #     next_state.set_current_value(all_farward(next_state.cumulative_choices, X_rnn))  ##加以后的值
            # else:
            #     next_state.set_current_value(None)  ##加以后的值
            next_state.set_current_round_index(self.current_round_index + 1)  ##长度，ex[-1,2,-1] 长度为3
            return next_state

    def r2(EEe, yy_1):
        yy_1 = yy_1.clone().detach().cpu().numpy()
        EEe = EEe.clone().detach().cpu().numpy()
        return 1 - (np.sum((yy_1 - EEe) ** 2)) / (np.sum((yy_1 - np.mean(yy_1)) ** 2))

    def Arity(s):
        if s in ['var_x1', 'var_x2', 'var_x3','var_x4','var_x5','var_x6','var_x7','var_x8','var_x9','c']:
            return 0
        if s in ['sin', 'cos', 'exp', 'ln', 'sqrt']:
            return 1
        if s in ['+', '-', '*', '/', '^']:
            return 2
    def softmax(x,c=100):
        return np.exp(c*x)/(np.sum(np.exp(c*x)))

    def best_child(node, is_exploration):  # 若子节点都扩展完了，求UCB值最大的子节点
        best_score = -sys.maxsize
        best_sub_node = None
        for k in range(len(operator_list)):
            # print('k',int(k))
            sub_node = node.get_children()[k]
            # print('sub_node',sub_node.get_state().cumulative_choices)
            # print('sub_node_visit', sub_node.get_visit_times())
            if is_exploration:
                C = 1 / math.sqrt(2.0) * 2
            else:
                C = 0.0
            left = sub_node.get_quality_value() / (sub_node.get_visit_times() + 0.0000001)
            if 1:
                # print('node.get_visit_times',node.get_visit_times())
                # right = 2.0 * np.sqrt(math.log(node.get_visit_times() + 0.0000001) / (sub_node.get_visit_times()+0.0000001))  ##论文
                # if node != None:
                #     right = 4/len(operator_list) * np.sqrt(node.get_visit_times()) / (1 + sub_node.get_visit_times())  ##论文
                # else:
                #     right = 4/len(operator_list) * np.sqrt(1) / (1 * sub_node.get_visit_times())  ##论文

                right = 2.0 * 1 / (0.00001 + 4 * sub_node.get_visit_times())  ##论文
            # else:
            #     right = 2.0 * np.sqrt(math.log(node.get_visit_times()) / (sub_node.get_visit_times()+0.0000001))

            # if node.parent != None:
            #     right=2.0*math.log(node.parent.get_visit_times())/sub_node.get_visit_times()
            # else:
            #     right = 2.0*math.log(node.get_visit_times()) / sub_node.get_visit_times()
            # print('mcts_output'*5,mcts_output)
            # print('mcts_output'*5,mcts_output_s)
            score = left + C * (right) * mcts_output[k]
            # print('score',score)
            if score > best_score:
                best_score = score
                best_sub_node = sub_node
                # print("left",left)
                # print('right',right)
        # print('best_sub_node',best_sub_node.get_state().cumulative_choices)
        return best_sub_node
    def expand(node):  # 得到未扩展的子节点
        for i in range(len(operator_list)):
            tried_sub_node_states = [sub_node.get_state() for sub_node in node.get_children()]
            # print('tried_sub_node_states',tried_sub_node_states)
            new_state = node.get_state().get_next_state_with_random_choice(operator_list[i])

            while new_state in tried_sub_node_states:
                print('a' * 200)
                new_state = node.get_state().get_next_state_with_random_choice(operator_list)  ##为了不前后两次都选择同一个数字。
                # print(new_state.get_state().get_next_state_with_random_choice().cumulative_choices)
            sub_node = Node()
            sub_node.set_state(new_state)
            node.add_child(sub_node)
        return node

    def backup(node, reward):
        while node != None:
            node.visit_times_add_one()  ##访问次数加一
            node.quality_value_add_n(reward)
            # node.set_quality_value(reward)
            node = node.parent  ##递归

    def farward_best(l2, X, X2, d):
        global index_c
        index_c = 0

        stack1 = []
        for i in range(len(l2)):
            # print(l2)
            s = l2[-(i + 1)]
            if s == 'var_x1':
                stack1.append(X)
            if s == 'var_x2':
                stack1.append(X2)
            if s == 'c':
                stack1.append(d[index_c])
                index_c = index_c + 1
            if s in ['sin', 'cos', 'log', 'exp', 'sqrt']:
                if s == 'exp':
                    v1 = np.exp(stack1.pop())
                if s == 'ln':
                    v1 = np.log(stack1.pop())
                if s == 'cos':
                    v1 = np.cos(stack1.pop())
                if s == 'sin':
                    v1 = np.sin(stack1.pop())
                if s == 'sqrt':
                    v1 = np.sqrt(stack1.pop())
                stack1.append(v1)
            if s in ['+', '-', '*', '/']:
                if s == '+':
                    v1 = stack1.pop() + stack1.pop()
                if s == '-':
                    v1 = stack1.pop() - stack1.pop()
                if s == '*':
                    v1 = stack1.pop() * stack1.pop()
                if s == '/':
                    v1 = stack1.pop() / stack1.pop()
                stack1.append(v1)
            # print('stack1',stack1)
        return stack1[0]

    epoch_best_rewards = []
    epoch_best_expressions = []

    epoch_mean_length = []
    epoch_mean_dl = []
    total_expr_lengths = []
    # Establish GPU device if necessary
    if (use_gpu and torch.cuda.is_available()):
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # load config prior
    config_prior = load_config(config_path=config_prior)["prior"]

    # Initialize operators, RNN, and optimizer
    operators = Operators(operator_list, device)
    prior = make_prior(library=operators, config_prior=config_prior)
    # print('AAAAAAAA',operators)
    dsr_rnn = DSRRNN(operators, hidden_size, device, min_length=min_length,
                     max_length=max_length, type=type, dropout=dropout, prior=prior).to(
        device)

    if (optimizer == 'adam'):
        optim = torch.optim.Adam(dsr_rnn.parameters(), lr=lr)
    else:
        optim = torch.optim.RMSprop(dsr_rnn.parameters(), lr=lr)

    # Best expression and its performance
    best_expression, best_performance = None, float('-inf')

    nod = Node()
    nod.set_state(State())
    computation_budget = 10000
    best_r = -sys.maxsize

    sequences = torch.zeros((1, 0))
    entropies = torch.zeros((1, 0))  # Entropy for each sequence
    log_probs = torch.zeros((1, 0))  # Log probability for each token


    # Order of observations: action, parent, sibling, dangling
    initial_obs = torch.tensor([dsr_rnn.operators.EMPTY_ACTION,
                                dsr_rnn.operators.EMPTY_PARENT,
                                dsr_rnn.operators.EMPTY_SIBLING,
                                1], dtype=torch.float32)
    initial_obs = initial_obs.repeat(1, 1)  # [batch_size, obs_dim]
    obs = initial_obs
    initial_prior = torch.from_numpy(dsr_rnn.prior.initial_prior())
    initial_prior = initial_prior.repeat(1, 1)  # [batch_size, n_choices]
    prior = initial_prior
    m_inupt = dsr_rnn.get_tensor_input(initial_obs)
    input_tensor = dsr_rnn.get_tensor_input(initial_obs)  # [batch_size, n_parent_inputs + n_sibling_inputs]

    hidden_tensor = dsr_rnn.init_hidden.repeat(1, 1)  # [batch_size, hid_dim]

    # First sampling done outside of loop for initial batch size if desired
    start = time.time()
    key = 0
    print(operator_list)
    print("2")

    def pip(rnode, rpi, pp, PP):
        vn = []
        all_children = rnode.get_children()
        if all_children != []:
            print('child'*20)
            for child in all_children:
                vn.append(child.get_visit_times())
            print('vn',vn)
        vn = np.array(vn)
        # if np.sum(vn) >= 2 * len(operator_list):
        if np.sum(vn) >= 2:
            print('vn'*20)
            rpi.append(softmax(vn, 1))
            pp.append(PP)
    st = 0
    N_var = 0
    for v in operator_list:
        if 'var' in v:
            N_var += 1
    index_x1 = operator_list.index('var_x1')
    for i in range(computation_budget):
        real_pi = [np.zeros(len(operator_list))]
        pred_p = [np.zeros(len(operator_list))]

        st = st + 1
        input_tensor = m_inupt

        print('LOOP FOR MCTS : ' + str(i))
        # print('input_tensor 1', input_tensor)

        if (dsr_rnn.type == 'lstm'):
            hidden_lstm = dsr_rnn.init_hidden_lstm.repeat(1, 1)
        if (dsr_rnn.type == 'rnn'):
            output, hidden_tensor = dsr_rnn.forward(input_tensor, hidden_tensor)
        elif (dsr_rnn.type == 'lstm'):
            output, hidden_tensor, hidden_lstm = dsr_rnn.forward(input_tensor, hidden_tensor, hidden_lstm)
        elif (dsr_rnn.type == 'gru'):
            output, hidden_tensor = dsr_rnn.forward(input_tensor, hidden_tensor)
        # print('output', output)
        mcts_output = softmax(output.detach().numpy()[0],c=10)
        # print('mcts_output',mcts_output)
        sorted_id = sorted(range(len(mcts_output)), key=lambda x: mcts_output[x], reverse=False)

        # expend_node = tree_policy(nod,dsr_rnn)

        expend_node = nod
        pip(expend_node,real_pi,pred_p,mcts_output)
        while expend_node.get_state().is_terminal() == False:
            # print(node.get_state().counter)
            if expend_node.is_all_expand():  ##如果已经扩展完毕
                # print('z'*100)
                expend_node = best_child(expend_node, True)
                pip(expend_node, real_pi, pred_p, mcts_output)
                while expend_node.get_visit_times() != 0 and expend_node.get_state().counter != 0:
                    # print('o' * 100)
                    if len(expend_node.get_children()) != 0:  ## Have expended

                        s_mcts = expend_node.get_state().cumulative_choices
                        # print('s_mcts',s_mcts[0:-1])
                        if s_mcts == []:
                            input_tensor = m_inupt
                        else:
                            MCTS_sequences = torch.zeros((1, 0))
                            for j in s_mcts:
                                aa = operator_list.index(j) * torch.ones(1)
                                MCTS_sequences = torch.cat((MCTS_sequences, aa[:, None]), axis=1)
                            if len(MCTS_sequences[0])>=10:
                                break
                            # Compute next parent and sibling; assemble next input tensor
                            next_obs, next_prior = dsr_rnn.get_next_obs(MCTS_sequences, obs)
                            next_obs, next_prior = torch.from_numpy(next_obs), torch.from_numpy(next_prior)
                            # print('MCTS_sequences',MCTS_sequences)
                            input_tensor = dsr_rnn.get_tensor_input(next_obs)
                            # print('input_tensor',input_tensor)
                            prior = next_prior
                            obs = next_obs

                            # print('input_tensor-11',input_tensor)
                        if (dsr_rnn.type == 'lstm'):
                            hidden_lstm = dsr_rnn.init_hidden_lstm.repeat(1, 1)
                        if (dsr_rnn.type == 'rnn'):
                            output, hidden_tensor = dsr_rnn.forward(input_tensor, hidden_tensor)
                        elif (dsr_rnn.type == 'lstm'):
                            output, hidden_tensor, hidden_lstm = dsr_rnn.forward(input_tensor, hidden_tensor,
                                                                                 hidden_lstm)
                        elif (dsr_rnn.type == 'gru'):
                            output, hidden_tensor = dsr_rnn.forward(input_tensor, hidden_tensor)
                        # print('output'*10, output)
                        mcts_output = softmax(output.detach().numpy()[0],c=10)
                        if s_mcts!= [] and 'c' in operator_list: ####determine wehther the seat of C is legal
                            if s_mcts[-1] in ['c','ln','sin','cos','exp','sqrt']:
                                mcts_output[operator_list.index('c')] = -100
                        expend_node = best_child(expend_node, True)
                        pip(expend_node, real_pi, pred_p, mcts_output)
                    else:
                        expend_node = expand(expend_node)
                        # node = best_child(node, True)
                # print('break' * 20)
                break
            else:
                # print('x' * 100)
                expend_node = expand(expend_node)  ##如果没有扩展完

        print('real_pi',real_pi)
        print('pred_p',pred_p)
        loss_real_pi = np.array(real_pi)
        loss_pred_p = np.array(pred_p)
        if st%20 == 0:
            loss_real_pi = np.array(real_pi)
            loss_pred_p = np.array(pred_p)
        print('MCTS', expend_node.get_state().cumulative_choices)
        MS = expend_node.get_state().cumulative_choices[0]
        if MS[0] == 'c':
            backup(expend_node, -10e10)
            continue
        # print('expend_node.get_state().counter',expend_node.get_state().counter)
        # print('expend_node.get_visit_times', expend_node.get_visit_times())


        # First sampling done outside of loop for initial batch size if desired

        # print('expend_node.get_state().cumulative_choices : ',expend_node.get_state().cumulative_choices)
        MT_node = expend_node.get_state().cumulative_choices
        # MT_node = ['cos', 'var_x']
        print('MT_node', MT_node)


        MT_cou = expend_node.get_state().counter
        # print('MT_cou',MT_cou)
        # initial_batch_size  = int(int(len(MT_node))*50) + 50
        initial_batch_size = int(int(i / len(operator_list) )** 2 + 100)
        # initial_batch_size = 1000 - int(int(i / len(operator_list)) ** 2 )
        initial_batch_size = min(initial_batch_size,1000)
        initial_batch_size = 200
        batch_size = initial_batch_size
        print('initial_batch_size', initial_batch_size)

        num_batches = 40 - int(int(i / len(operator_list)) ** 2 )
        num_batches = max(num_batches,4)
        num_batches = 2
        print('num_batches', num_batches)
        # initial_batch_size  = 2000 - int(int(i/10)**2)

        sequences, sequence_lengths, log_probabilities, entropies = dsr_rnn.sample_n_expressions(initial_batch_size,MT_node,MT_cou,operator_list)  ##initial_batch_size ：The number of sampled expression.
        # sequences, sequence_lengths, log_probabilities, entropies = dsr_rnn.sample_sequence(initial_batch_size) ##initial_batch_size ：The number of sampled expression.
        batch_best = -np.inf
        for i in range(num_batches):
            # Convert sequences into Pytorch expressions that can be evaluated
            #### all constants
            # Seq = []
            # for j in range(len(sequences)):
            #     l = sequences[j]
            #     L = []
            #     for i in range(len(l)):
            #         lb = [operator_list.index('+'), operator_list.index('*'), l[i], operator_list.index('c'), operator_list.index('c')]
            #         for k in lb:
            #             L.append(k)
            #     Seq.append(L)
            # sequences = torch.tensor(Seq)
            # sequence_lengths = 5 * sequence_lengths
            ###################
            expressions = []
            for j in range(len(sequences)):
                expressions.append(
                    Expression(operators, sequences[j].long().tolist(), sequence_lengths[j].long().tolist()).to(device)
                )

            # 计算epoch的平均表达式长度
            epoch_mean_length.append(torch.mean(sequence_lengths.float()).item())
            total_expr_lengths.append(sequence_lengths.float())
            # Optimize constants of expressions (training data)
            best_reward = [-np.inf]
            expressions_2 = (expressions).copy()
            # print('expressions_2',expressions[0])
            print('expressions',expressions[0])

            for p in range(1):
                optimize_constants(expressions, X_constants, y_constants, inner_lr, inner_num_epochs, inner_optimizer)

                # Benchmark expressions (test dataset)
                rewards = []
                for ep in range(len(expressions)):
                    # print('sequence_lengths',int(sequence_lengths[ep]))
                    rewards.append(benchmark(expressions[ep], X_rnn, y_rnn, sequences[ep],int(sequence_lengths[ep]), N_var, index_x1))
                if max(rewards)>=max(best_reward):
                    best_reward = rewards

            rewards = torch.tensor(best_reward)


            # Update best expression

            best_epoch_expression = expressions[np.argmax(rewards)]
            epoch_best_expressions.append(best_epoch_expression)
            epoch_best_rewards.append(max(rewards).item())
            best_epoch_seq = sequences[np.argmax(rewards)]
            if (max(rewards) > best_performance):
                print('best_performance',best_performance)
                best_performance = max(rewards)
                best_expression = best_epoch_expression
                # print('best_expression',best_expression)
                best_seq = best_epoch_seq
                # print('best_seq',best_seq)
                best_seq_l = sequence_lengths[np.argmax(rewards)]

            if (max(rewards) > batch_best):
                batch_best = max(rewards)
                batch_best_expression = best_epoch_seq

            # Early stopping criteria
            if (best_performance >= 0.9999):
                best_str = str(best_expression)
                if (live_print):
                    print("~ Early Stopping Met ~")
                    print(f"""Best Expression: {best_str}""")
                break

            # #### optimize the best one loop
            # best_list = []
            # best_list.append(Expression(operators, best_seq.long().tolist(), best_seq_l.long().tolist()).to(device))
            # # print('best_expression_2', best_list[0])
            # optimize_constants(best_list, X_constants, y_constants, inner_lr, inner_num_epochs, inner_optimizer)
            # reward_2 = benchmark(best_list[0], X_rnn, y_rnn)
            # # print('best_list',best_list[0])
            # print('rewards_2',reward_2)

            #### optimize the best one loop

            # print('best_seq-1',best_seq)
            # print('best_seq_l',best_seq_l)
            best_list = []
            best_list.append(
            Expression(operators, best_seq.long().tolist(), best_seq_l.long().tolist()).to(device)
            )
            # print(best_list[0])
            if len(best_list) != 0:
                print('z'*100)
                optimize_constants(best_list, X_constants, y_constants, inner_lr, inner_num_epochs, inner_optimizer)
                reward_2 = benchmark(best_list[0], X_rnn, y_rnn, best_seq, best_seq_l,N_var, index_x1)
                # print('best_list',best_list[0])
            print(reward_2)

            # l2 = []
            # print(best_seq)
            # best_seq_counter = 1
            # for j in range(len(best_seq)):
            #     l2.append(operator_list[int(best_seq[j])])
            #     best_seq_counter = best_seq_counter + Arity(operator_list[int(best_seq[j])]) - 1
            #     if best_seq_counter == 0:
            #         break
            #
            # XX1 = X_constants[:,0].numpy()
            # XX2 = X_constants[:,1].numpy()
            # y = y_constants.numpy()
            # print(l2)
            # fun_loss = lambda D : np.mean((y -  farward_best(l2,XX1,XX2,D))**2)
            # number_c = len(np.argwhere(np.array(l2) == 'c'))
            # x0 = np.random.randn(number_c)
            #
            # print(x0)
            # reward_2 = -np.inf
            # if len(x0) > 0 and 1:
            #     for i in range(1):
            #         print("z"*100)
            #         res = minimize(fun_loss, x0, method="BFGS")
            #         print('resx',res.x)
            #     # print(farward_best(l2,np.array(X_rnn[:,0]),np.array(X_rnn[:,1]),res.x))
            #     reward_2 = reward_nrmse(torch.tensor(farward_best(l2,np.array(X_rnn[:,0]),np.array(X_rnn[:,1]),res.x)), y_rnn)
            #     print(reward_2)

            if (reward_2 > best_performance):
                print('best_performance',best_performance)
                best_performance = torch.tensor(reward_2)
                best_expression = best_list[0]

            if reward_2 >= 0.9999:

                best_str = str(best_expression)
                if (live_print):
                    print("~ Early Stopping Met ~")
                    print(f"""Best Expression: {best_str}""")
                break


            # Compute risk threshold
            if (i == 0 and scale_initial_risk):
                threshold = np.quantile(rewards, 1 - (1 - risk_factor) / (initial_batch_size / batch_size))
            else:
                threshold = np.quantile(rewards, risk_factor)
            indices_to_keep = torch.tensor([j for j in range(len(rewards)) if rewards[j] > threshold])

            if (len(indices_to_keep) == 0 and summary_print):
                print("Threshold removes all expressions. Terminating.")
                break
                # continue
            # Select corresponding subset of rewards, log_probabilities, and entropies
            rewards = torch.index_select(rewards, 0, indices_to_keep)
            log_probabilities = torch.index_select(log_probabilities, 0, indices_to_keep)
            entropies = torch.index_select(entropies, 0, indices_to_keep)

            # Compute risk seeking and entropy gradient
            risk_seeking_grad = torch.sum((rewards - threshold) * log_probabilities, axis=0)
            entropy_grad = torch.sum(entropies, axis=0)

            # Mean reduction and clip to limit exploding gradients
            risk_seeking_grad = torch.clip(risk_seeking_grad / len(rewards), -1e6, 1e6)
            entropy_grad = entropy_coefficient * torch.clip(entropy_grad / len(rewards), -1e6, 1e6)

            #Compute loss and backpropagate
            # loss = 1 * -1 * lr * (risk_seeking_grad + entropy_grad) + 1 * np.sum((loss_pred_p - loss_real_pi)**2)/len(operator_list)
            # print(loss_real_pi)  #.detach().numpy()
            loss_r = torch.tensor(loss_real_pi[0])
            loss_p = torch.tensor(loss_pred_p[0])
            # print(loss_r)
            # print(loss_p)
            print('log : ',torch.mean(loss_r * torch.log(loss_p.T + 0.001)))
            # loss = 1 * -1 * lr * (risk_seeking_grad + entropy_grad) + 1 * torch.mean((loss_r - loss_p)**2)
            loss = 1 * -1 * lr * (entropy_grad) + 1 * torch.mean(loss_r * torch.log(loss_p.T + 0.001)) - lr * (entropy_grad)
            loss.requires_grad_(True)
            loss.backward()
            optim.step()

            # Sample for next batch
            # sequences, sequence_lengths, log_probabilities, entropies = dsr_rnn.sample_sequence(batch_size)
            sequences, sequence_lengths, log_probabilities, entropies = dsr_rnn.sample_n_expressions(batch_size,MT_node,MT_cou,operator_list)  ##initial_batch_size ：The number of sampled expression.

        backup(expend_node, batch_best / 1)
        # batch_best = -np.inf
        if (summary_print):
            print(f"""
            Time Elapsed: {round(float(time.time() - start), 2)}
            Epochs Required: {i+1}
            Best Performance: {round(best_performance.item(),4)}
            Best Expression: {best_expression}
            Target Expression:{'  y =  greenhouse '}
            """)
        if (best_performance >= 0.9999):
            best_str = str(best_expression)
            if (live_print):
                print("~ Early Stopping Met ~")
                print(f"""Best Expression: {best_str}""")
            break

    return [epoch_best_rewards, epoch_best_expressions, best_performance, best_expression]

###############################################################################
# Reward function
###############################################################################

def benchmark(expression, X_rnn, y_rnn, SEQ, LEN, Nvar,indx):
    """Obtain reward for a given expression using the passed X_rnn and y_rnn
    """
    with torch.no_grad():
        y_pred = expression(X_rnn)
        C = 0.1
        loss_x = 0.
        for i in range(Nvar):
            X_pred = torch.zeros(len(X_rnn[:, 0]))
            if indx+i not in SEQ[0: LEN]:
                # X_pred = torch.tensor(X_rnn[:, i])
                loss_x = loss_x + reward_nrmse(X_pred, X_rnn[:, 0])

        rew = reward_nrmse(y_pred, y_rnn) + C * loss_x
        # X1_pred = torch.zeros(len(X_rnn[:,0]))
        # X2_pred = torch.zeros(len(X_rnn[:,1]))
        #
        # if 6 in SEQ[0: LEN]:
        #     X1_pred = torch.tensor(X_rnn[:,0])
        # if 7 in SEQ[0: LEN]:
        #     X2_pred = torch.tensor(X_rnn[:,1])
        # # print('X2_pred', X2_pred)
        # # print('X_rnn[:,1]', X_rnn[:,1])
        # # print('reward_nrmse(X2_pred,X_rnn[:,0])',reward_nrmse(X2_pred,X_rnn[:,1]))
        # rew = reward_nrmse(y_pred, y_rnn) + C * reward_nrmse(X1_pred,X_rnn[:,0]) + C * reward_nrmse(X2_pred,X_rnn[:,1])
        return 1/(1+rew)

def reward_nrmse(y_pred, y_rnn):
    """Compute NRMSE between predicted y and actual y
    # """
    # loss = nn.MSELoss()
    # val = torch.sqrt(loss(y_pred, y_rnn)) # Convert to RMSE
    # val = torch.std(y_rnn) * val # Normalize using stdev of targets
    # # print('val',val)
    # val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10)) # Fix nan and clip
    # val = 1 / (1 + val) # Squash
    # # val = np.tanh(val)
    # return val.item()

    # loss = nn.MSELoss()
    # val = loss(y_pred, y_rnn) # Convert to RMSE
    # val = torch.std(y_rnn) * val  # Normalize using stdev of targets
    # val =torch.sqrt(val)
    # # print('val',val)
    # val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip
    # val = 1 / (1 + val)  # Squash
    # # val = np.tanh(val)
    # return val.item()

    #### S_NRMSE
    loss = nn.MSELoss()
    val = loss(y_pred, y_rnn)  # Convert to RMSE
    val = torch.std(y_rnn) * val  # Normalize using stdev of targets
    val = torch.sqrt(val)
    # print('val',val)
    val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip

    # val = 1 / (1 + val)  # Squash
    # val = np.tanh(val)
    return val.item()

    #
    # loss = (y_pred - y_rnn)**2
    # # Min-Max scaling
    # min_a = torch.min(loss)
    # max_a = torch.max(loss)
    # loss =100 *  (loss - min_a) / (max_a - min_a)
    # val = torch.sqrt(torch.mean(loss))  # Convert to RMSE
    # # val = torch.std(y_rnn) * val  # Normalize using stdev of targets
    # val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip
    # val = 1 / (1 + val)  # Squash
    # return val.item()

    # val = torch.relu(1 - (torch.mean((y_pred - y_rnn) ** 2)/torch.mean((y_rnn - torch.mean(y_rnn))**2)))
    # print('val',val)
    # val = min(torch.nan_to_num(val, nan=-0.), torch.tensor(-0.))  # Fix nan and clip
    # return val.item() #####end
