
[//]: # (# Discovering mathematical formulas from data via LSTM-guided Monte Carlo Tree Search)
<h1 align="center">Discovering mathematical formulas from data via LSTM-guided Monte Carlo Tree Search
</h1>
<h2 align="center">

[![Mentioned in Awesome Vue.js](https://awesome.re/mentioned-badge.svg)](https://anonymous.4open.science/r/AlphaSymbol-v2)

[//]: # ([![Download my paper Vue.js]&#40;https://img.shields.io/badge/vue-2.2.4-green.svg&#41;]&#40;https://github.com/vuejs/awesome-vue&#41;)


</h2>

<p align="center">
<img src="https://img.shields.io/badge/Symbolic-Regression-brightgreen" >
<img src="https://img.shields.io/badge/Download%20-AlphaSymbol-yellow">
<img src="https://img.shields.io/badge/Watching-0-green">

[//]: # (<img src="https://img.shields.io/badge/vue-2.2.4-green.svg">)

<img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" >

[//]: # (<img src="https://beerpay.io/silent-lad/VueSolitaire/badge.svg?style=flat">)

<img src="https://img.shields.io/badge/Start-0-blue">

<img src="https://img.shields.io/badge/Python-100%25-green">

<img src="https://img.shields.io/badge/forks-0-yellow">

[//]: # (<img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat">)
</p>
<div align=center>
<img src="figure/AlphaSymbol8.png" alt="Image" width="80%" height="100%">
</div>



[//]: # (This repository is the official implementation of [My Paper Title]&#40;https://arxiv.org/abs/2030.12345&#41;. )

## Description
It is an important purpose of scientific research to find a formula to describe the operation of the physical world from observational data. In this paper, we propose a method, AlphaSymbol, which can achieve this goal without human experience.
Specifically, we use the "collaboration" between LSTM and MCTS algorithms to efficiently discover mathematical formulas from data. Compared with the traditional machine learning algorithm, the result obtained by AlphaSymbol is a mathematical formula, which can not only fit the data well, but also be concise and analyzable.
For example, in the physical formula $\mathcal{P = FV}$, we can easily analyze that, given a constant power $\mathcal{P}$, to obtain a larger force $\mathcal{F}$, we must decrease the velocity $\mathcal{V}$. This is also why vehicles slow down when going uphill in real-life scenarios. However, a model obtained from a traditional machine learning algorithm cannot intuitively derive such useful conclusions. 

## Requirements

(1),Create a Python 3.7 virtual environment
```setup
conda create -n venv python=3.7 # Create a Python 3.7 virtual environment
```
(2),To activate the newly created environment, use the following command:
```setup
source activate venv  # activate the newly created environment
```
(3),To install requirements:
```setup
pip install -r requirements.txt # To install all the required dependencies. 
```

[//]: # (>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...)

## Getting started

(1), Setting the appropriate parameters in the `main.py`  file
```js
results = train(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        # operator_list = S,
        operator_list = ['*','+','-','/','sin','cos','var_x1'],
        min_length = 2,
        max_length = 60,
        type = 'lstm',
        num_layers = 4,
        hidden_size = 500,
        dropout = 0.0,
        lr = 0.0005,
        optimizer = 'adam',
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
```
(2), In the `config_prior.py` file, select the restrictions to apply during the search (true or false).
(Note that the maximum and minimum length restrictions in `config_prior.py` are the same as those in main.py.)
```js
{"prior": {
  "length": {
    "min_": 2,
    "max_": 20,
    "on": true
  },
  "repeat": {
    "tokens": "c",
    "min_": null,
    "max_": 11,
    "on": true
  },
  "inverse": {
    "on": true
  },
  "trig": {
    "on": true
  },
  "const": {
    "on": true
  },
  "no_inputs": {
    "on": true
  },
  "uniform_arity": {
    "on": false
  },
  "soft_length": {
    "loc": 10,
    "scale": 5,
    "on": false
  }
 }
}
```
(3), Start finding formulas
```train
python main.py 
```

[//]: # ()
[//]: # (>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. )

[//]: # (> 在main.py文件里面指定算法的各种参数，然后指定对应的X和Y，)
## Evaluation

  We evaluated the performance of AlphaSymbol on more than ten classic datasets in the field of symbolic regression. These datasets are labeled **Nguyen, Keijzer, Korns, Constant, Livermore, R, Vladislavlev, Jin, Neat, AI Feynman, and Others**. The datasets mentioned above collectively contain a total of 222 test expressions.
We compare AlphaSymbol with four symbol regression algorithms that have demonstrated high performance:

- **DSO**. A superior algorithm that effectively integrates reinforcement learning and genetic programming (GP) for symbolic regression tasks. 

- **DSR**. An exceptional algorithm that effectively employs deep reinforcement learning in the domain of symbolic regression.

- **GP**. A classic algorithm that applies genetic algorithms perfectly to the field of symbolic regression. Greatly improved search efficiency.

- **NeSymReS**. This algorithm is categorized as a large-scale pre-training model.

[//]: # (The Pythagorean theorem is $a^2 + b^2 = c^2.$)

[//]: # (```eval)

[//]: # (python eval.py --model-file mymodel.pth --benchmark imagenet)

[//]: # (```)
[//]: # (## Pre-trained Models)

[//]: # ()
[//]: # (You can download pretrained models here:)

[//]: # ()
[//]: # (- [My awesome model]&#40;https://drive.google.com/mymodel.pth&#41; trained on ImageNet using parameters x,y,z. )

## Results

[//]: # (### [Image Classification on ImageNet]&#40;https://paperswithcode.com/sota/image-classification-on-imagenet&#41;)

[//]: # (| Model name         | Top 1 Accuracy  | Top 5 Accuracy |)

[//]: # (| ------------------ |---------------- | -------------- |)

[//]: # (| My awesome model   |     85%         |      95%       |)

Recovery rate comparison of AlphaSymbol and four baselines on more than ten mainstream symbolic regression datasets.

<img src="figure/table1.png" alt="Image" width="70%" height="60%">


Average Coefficient of Determination ($R^2$) on Various Datasets

<img src="figure/r2.png" alt="Image" width="20%" height="20%">


This figure describes the recovery rate of AlphaSymbol and four other excellent algorithms on all Nguyen benchmarks under different levels of noise.

[//]: # (![Sample results plot]&#40;noise.png&#41;)
<img src="figure/noise.png" alt="Image" width="60%" height="60%">


[//]: # (## Contributing)

[//]: # ()
[//]: # (>📋  Pick a licence and describe how to contribute to your code repository. )