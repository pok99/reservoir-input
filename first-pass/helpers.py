
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import random

def get_optimizer(args, train_params):
    op = None
    if args.optimizer == 'adam':
        op = optim.Adam(train_params, lr=args.lr)
    elif args.optimizer == 'sgd':
        op = optim.SGD(train_params, lr=args.lr)
    elif args.optimizer == 'rmsprop':
        op = optim.RMSprop(train_params, lr=args.lr)
    elif args.optimizer == 'lbfgs-pytorch':
        op = optim.LBFGS(train_params, lr=0.75)
    return op

def get_criterion(args):
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    return criterion

def get_output_activation(args):
    if args.out_act == 'exp':
        fn =  torch.exp
    elif args.out_act == 'relu':
        fn =  nn.ReLU()
    elif args.out_act == 'none':
        fn =  lambda x: x
    return fn