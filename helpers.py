
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pdb

import random

def sigmoid(x):
    return 1/(1 + np.exp(-x))

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

def get_criteria(args):
    criteria = []
    if 'mse' in args.losses:
        fn = nn.MSELoss(reduction='sum')
        def mse(t, o, i=None):
            return args.mse_l1 * fn(t, o)
        criteria.append(mse)
    if 'bce' in args.losses:
        weights = args.bce_l2 * torch.ones(1)
        fn = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=weights)
        def bce(t, o, i=None):
            return args.bce_l1 * fn(t, o)
        criteria.append(bce)
    if 'mse-w' in args.losses:
        fn = nn.MSELoss(reduction='sum')
        def mse_w(t, o, i):
            loss = 0.
            if len(o.shape) == 1:
                o = o.unsqueeze(0)
                t = t.unsqueeze(0)
                i = [i]
            for j in range(len(t)):
                t_set, t_go = i[j][0], i[j][2]
                # using interval from t_set to t_go + t_p
                loss += fn(t[j,t_set:2*t_go-t_set+1], o[j,t_set:2*t_go-t_set+1])
            return args.mse_l1 * loss
        criteria.append(mse_w)
    if 'bce-w' in args.losses:
        weights = args.bce_l2 * torch.ones(1)
        fn = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=weights)
        def bce_w(t, o, i):
            loss = 0.
            if len(o.shape) == 1:
                o = o.unsqueeze(0)
                t = t.unsqueeze(0)
                i = [i]
            for j in range(len(t)):
                t_set, t_go = i[j][0], i[j][2]
                # using interval from t_set to t_go + t_p
                loss += fn(t[j,t_set:2*t_go-t_set+1], o[j,t_set:2*t_go-t_set+1])
            return args.bce_l1 * loss
        criteria.append(bce_w)
    if len(criteria) == 0:
        raise NotImplementedError
    return criteria

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
    if len(outs.shape) == 1:
        x = x.unsqueeze(0)
        outs = outs.unsqueeze(0)
        info = [info]
    for j in range(len(x)):
        # pdb.set_trace()
        ready, go = info[j][0], info[j][2]
        # getting the index of the first timestep where output is above threshold
        first_t = torch.nonzero(outs[j][ready:] > l1)
        if len(first_t) == 0:
            first_t = torch.tensor(len(x[j])) - 1
        else:
            first_t = first_t[0,0] + ready
        targets = None
        # losses defined on interval b/w go and first_t
        if go > first_t:
            relevant_outs = outs[j][first_t:go+1]
            targets = torch.zeros_like(relevant_outs)
            weights = torch.arange(go+1-first_t,0,-1)
        elif go < first_t:
            relevant_outs = outs[j][go:first_t + 1]
            targets = l1 * torch.ones_like(relevant_outs)
            weights = torch.arange(0,first_t+1-go,1)
        first_ts.append(first_t)
        if targets is not None:
            mse2_loss = torch.sum(weights * torch.square(targets - relevant_outs))
            # mse2_loss = diff * nn.MSELoss(reduction='sum')(targets, relevant_outs)
            total_loss += mse2_loss
    if extras:
        first_t_avg = sum(first_ts) / len(first_ts)
        return l2 * total_loss, first_t_avg
    return l2 * total_loss