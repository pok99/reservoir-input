
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pdb

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
        # criterion = nn.MSELoss()
        criterion = nn.MSELoss(reduction='sum')
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'mse2':
        criterion = nn.MSELoss(reduction='sum')

    return criterion

def get_output_activation(args):
    if args.out_act == 'exp':
        fn = torch.exp
    elif args.out_act == 'relu':
        fn = nn.ReLU()
    elif args.out_act == 'none':
        fn = lambda x: x
    return fn

# given batch, get the x, y pairs and turn them into Tensors
def get_x_y_info(batch):
    x, y, info = list(zip(*batch))
    x = torch.as_tensor(x, dtype=torch.float)
    y = torch.as_tensor(y, dtype=torch.float)
    return x, y, info

def get_dim(a):
    if hasattr(a, '__iter__'):
        return len(a)
    else:
        return 1

def mse2_loss(x, outs, info, l1, l2, extras=False):
    total_loss = 0.
    first_ts = []
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
        outs = outs.unsqueeze(0)
        info = [info]
    for j in range(len(x)):
        ready, go = info[j][0], info[j][2]
        first_t = torch.argmax(outs[j][ready:]) + ready
        # first_t = torch.nonzero(outs[j][ready:] > l1)
        # if len(first_t) == 0:
        #     first_t = torch.tensor(len(x[j]))
        # else:
        #     first_t = first_t[0][0]
        first_ts.append(first_t)
        mse2_loss = torch.square(first_t - go)
        total_loss += mse2_loss
    if extras:
        first_t_avg = sum(first_ts) / len(first_ts)
        return l2 * total_loss, first_t_avg
    # pdb.set_trace()
    return l2 * total_loss