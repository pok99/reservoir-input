
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
        criterion = nn.MSELoss()
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'seq-goals':
        criterion = seq_goals_loss
    return criterion

def get_output_activation(args):
    if args.out_act == 'exp':
        fn =  torch.exp
    elif args.out_act == 'relu':
        fn =  nn.ReLU()
    elif args.out_act == 'none':
        fn =  lambda x: x
    return fn

# loss function for sequential goals
def seq_goals_loss(out, target, threshold=.3, reward=5):
    if len(out.shape) > 1:
        dists = torch.norm(out - target, dim=1)
    else:
        # just one dimension so only one element in batch
        dists = torch.norm(out - target, dim=0, keepdim=True)

    done = (dists < threshold) * 1
    done_count = done.sum()
    loss = torch.sum(dists) - done_count * reward

    return loss, done

# updating indices array to get the next targets for sequential goals
def update_seq_indices(targets, indices, done):
    n_pts = len(targets)
    n_trials = len(indices)
    for seq in range(n_trials):
        if indices[seq] < n_pts - 1 and done[seq]:
            indices[seq] += 1
    return indices

# given batch and dset name, get the x, y pairs
def get_x_y(batch, dset):
    if 'seq-goals' in dset:
        x = torch.Tensor(batch)
        y = x
    else:
        x, y, _ = list(zip(*batch))
        x = torch.Tensor(x)
        y = torch.Tensor(y)

    return x, y

def get_dim(a):
    if hasattr(a, '__iter__'):
        return len(a)
    else:
        return 1