###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip

# Original Paper: https://arxiv.org/abs/1912.04871 (Petersen et al)

# operators.py: Operator class, which is used as the building blocks for
# assembling PyTorch expressions with an RNN.

###############################################################################
# Dependencies
###############################################################################

import torch
import numpy as np
###############################################################################
# Operators Class
###############################################################################

class Operators:
    """
    The list of valid nonvariable operators may be found in nonvar_operators.
    All variable operators must have prefix 'var_'. Constant value operators
    are fine too (e.g. 3.14), but they must be passed as floats.
    """
    nonvar_operators = [
        '*', '+', '-', '/', '^',
        'cos', 'sin', 'tan',
        'exp', 'ln',
        'sqrt', 'square',
        'c' # ephemeral constant
    ]
    nonvar_arity = {
        '*': 2,
        '+': 2,
        '-': 2,
        '/': 2,
        '^': 2,
        'cos': 1,
        'sin': 1,
        'tan': 1,
        'exp': 1,
        'ln': 1,
        'sqrt': 1,
        'square': 1,
        'c': 0
    }
    function_mapping = {
        '*': 'torch.mul',
        '+': 'torch.add',
        '-': 'torch.subtract',
        '/': 'torch.divide',
        '^': 'torch.pow',
        'cos': 'torch.cos',
        'sin': 'torch.sin',
        'tan': 'torch.tan',
        'exp': 'torch.exp',
        'ln': 'torch.log',
        'sqrt': 'torch.sqrt',
        'square': 'torch.square'
    }

    def __init__(self, operator_list, device):
        """Description here
        """
        self.operator_list = operator_list
        self.constant_operators = [x for x in operator_list if x.replace('.', '').strip('-').isnumeric()]
        self.nonvar_operators = [x for x in self.operator_list if "var_" not in x and x not in self.constant_operators]
        self.var_operators = [x for x in operator_list if x not in self.nonvar_operators and x not in self.constant_operators]
        self.__check_operator_list() # Sanity check

        self.device = device

        # Construct data structures for handling arity
        self.arity_dict = dict(self.nonvar_arity, **{x: 0 for x in self.var_operators}, **{x: 0 for x in self.constant_operators})
        self.zero_arity_mask = torch.tensor([1 if self.arity_dict[x]==0 else 0 for x in self.operator_list]).to(device)
        self.nonzero_arity_mask = torch.tensor([1 if self.arity_dict[x]!=0 else 0 for x in self.operator_list]).to(device)
        self.variable_mask = torch.Tensor([1 if x in self.var_operators else 0 for x in self.operator_list])
        self.nonvariable_mask = torch.Tensor([0 if x in self.var_operators else 1 for x in self.operator_list])

        # Contains indices of all operators with arity 2
        self.arity_two = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==2])
        # Contains indices of all operators with arity 1
        self.arity_one = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==1])
        # Contains indices of all operators with arity 0
        self.arity_zero = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==0])
        # Contains indices of all operators that are variables
        self.variable_tensor = torch.Tensor([i for i in range(len(self.operator_list)) if operator_list[i] in self.var_operators])

        # Construct data structures for handling function and variable mappings
        self.func_dict = dict(self.function_mapping)
        self.var_dict = {var: i for i, var in enumerate(self.var_operators)}

        # 自定义三角函数和对指数约束mask
        self.non_sin_cos_mask = torch.Tensor(
            [0 if i in ['sin', 'cos'] else 1 for i in self.operator_list])  # sin cos的位置为0其他位置为1
        self.non_exp_ln_mask = torch.Tensor(
            [0 if i in ['ln', 'exp'] else 1 for i in self.operator_list])  # log exp的位置为0其他位置为1
        self.cos_sin_index = torch.Tensor(
            [i for i in range(len(self.operator_list)) if self.operator_list[i] in ['sin', 'cos']])
        self.exp_ln_index = torch.Tensor(
            [i for i in range(len(self.operator_list)) if self.operator_list[i] in ['exp', 'ln']])

        ######################################### 参考DSR添加约束需要用到的变量 ########################################
        self.arities = np.array([self.arity_dict[token] for token in self.operator_list], dtype=np.int32)
        self.parent_adjust = np.full_like(self.arities, -1)
        count = 0
        for i in range(len(self.arities)):
            if self.arities[i] > 0:
                self.parent_adjust[i] = count
                count += 1

        self.L = len(self.operator_list)
        self.terminal_tokens = self.arity_zero
        self.n_action_inputs = self.L + 1  # Library tokens + empty token
        self.n_parent_inputs = self.L + 1 - len(self.terminal_tokens)  # Parent sub-lib tokens + empty token
        self.n_sibling_inputs = self.L + 1  # Library tokens + empty token
        self.EMPTY_ACTION = self.n_action_inputs - 1
        self.EMPTY_PARENT = self.n_parent_inputs - 1
        self.EMPTY_SIBLING = self.n_sibling_inputs - 1

        # 相反约束需要用到的变量
        inverse_tokens = {
            "inv": "inv",
            "neg": "neg",
            "exp": "ln",
            "ln": "exp",
            "sqrt": "n2",
            "n2": "sqrt",
        }
        token_from_name = {t: i for i, t in enumerate(self.operator_list)}  # ['token': id]
        self.inverse_tokens = {token_from_name[k]: token_from_name[v] for k, v in inverse_tokens.items() if
                               k in token_from_name and v in token_from_name}

        # 三角函数嵌套需要用到的变量
        trig_names = ["sin", "cos"]
        self.trig_tokens = np.array([i for i, t in enumerate(self.operator_list) if t in trig_names], dtype=np.int32)
        self.binary_tokens = self.arity_two
        self.unary_tokens = self.arity_one

        # 常数个数约束需要用到的变量
        try:
            self.const_token = self.operator_list.index("c")
        except ValueError:
            self.const_token = None

        # 至少有一个自变量的约束需要用到的变量, 以后debug可以重点关注这里，可能会出错
        self.float_tokens = np.array(
            [i for i, f in enumerate(self.operator_list) if f.replace('.', '').strip('-').isnumeric()])  # 小数
        if self.const_token is not None:
            self.float_tokens += np.array([self.const_token])

        self.names = self.operator_list

    def __check_operator_list(self):
        """Throws exception if operator list is bad
        """
        invalid = [x for x in self.nonvar_operators if x not in Operators.nonvar_operators]
        if (len(invalid) > 0):
            raise ValueError(f"""Invalid operators: {str(invalid)}""")
        return True

    def __getitem__(self, i):
        # try:
        #     return self.operator_list[i]
        # except:
        #     return self.operator_list.index(i)
        if isinstance(i, (int, np.integer)):
            return self.operator_list[i]
        elif isinstance(i, str):
            try:
                return self.operator_list.index(i)
            except:
                raise TokenNotFoundError("Token {} does not exist.".format(i))
        else:
            raise TokenNotFoundError("Token {} does not exist.".format(i))

    def arity(self, operator):
        try:
            return self.arity_dict[operator]
        except NameError:
            print("Invalid operator")

    def arity_i(self, index):
        try:
            return self.arity_dict[self.operator_list[index]]
        except NameError:
            print("Invalid index")

    def func(self, operator):
        return self.func_dict[operator]

    def func_i(self, index):
        return self.func_dict[self.operator_list[index]]

    def var(self, operator):
        return self.var_dict[operator]

    def var_i(self, index):
        return self.var_dict[self.operator_list[index]]

    def __len__(self):
        return len(self.operator_list)
    def tokenize(self, inputs):
        """Convert inputs to list of Tokens."""

        if isinstance(inputs, str):
            inputs = inputs.split(',')
        elif not isinstance(inputs, list) and not isinstance(inputs, np.ndarray):
            inputs = [inputs]
        # tokens = [input_ if isinstance(input_, Token) else self[input_] for input_ in inputs]
        tokens = [input_ if input_ in self.operator_list else self[input_] for input_ in inputs]
        return tokens

    def actionize(self, inputs):
        """Convert inputs to array of 'actions', i.e. ints corresponding to
        Tokens in the Library."""

        tokens = self.tokenize(inputs)
        actions = np.array([self.operator_list.index(t) for t in tokens], dtype=np.int32)
        return actions
class TokenNotFoundError(Exception):
    pass