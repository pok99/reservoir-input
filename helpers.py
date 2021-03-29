
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pdb

import random

from trials import get_x, get_y
from utils import load_rb

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_optimizer(args, train_params):
    op = None
    if args.optimizer == 'adam':
        op = optim.Adam(train_params, lr=args.lr, weight_decay=args.l2_reg)
    elif args.optimizer == 'sgd':
        op = optim.SGD(train_params, lr=args.lr, weight_decay=args.l2_reg)
    elif args.optimizer == 'rmsprop':
        op = optim.RMSprop(train_params, lr=args.lr, weight_decay=args.l2_reg)
    elif args.optimizer == 'lbfgs-pytorch':
        op = optim.LBFGS(train_params, lr=0.75)
    return op

def get_scheduler(args, op):
    if args.s_rate is not None:
        return optim.lr_scheduler.MultiStepLR(op, milestones=[1,2,3], gamma=args.s_rate)
    return None

# dataset that automatically creates trials composed of trial and context data
class TrialDataset(Dataset):
    def __init__(self, datasets, args):
        self.datasets = datasets
        self.args = args
        # arrays of just the context cues
        self.x_contexts = []
        # cumulative lengths of datasets
        self.max_idxs = [0]
        for i, ds in enumerate(datasets):
            x_ctx = np.zeros((args.T, ds[0]['trial_len']))
            # setting context cue for appropriate task
            x_ctx[i] = 1
            self.x_contexts.append(x_ctx)
            if i != args.T - 1:
                self.max_idxs.append(self.max_idxs[i] + len(ds))

    def __len__(self):
        return self.max_idxs[-1]

    def __getitem__(self, idx):
        # index into the appropriate dataset to get the trial
        ds_id = np.argmax(self.max_idxs > idx)
        if ds_id == 0:
            trial = self.datasets[0][idx]
        else:
            trial = self.datasets[ds_id][idx - self.max_idxs[ds_id-1]]

        # combine context cue with actual trial x to get final x
        x = get_x(trial, self.args)
        x_cts = self.x_contexts[ds_id]
        x = np.concatenate((x, x_cts))
        # don't need to do that with y
        y = get_y(trial, self.args)
        trial['context'] = ds_id
        return x, y, trial

# turns data samples into stuff that can be run through network
def collater(samples):
    xs, ys, infos = list(zip(*samples))
    xs = torch.as_tensor(np.stack(xs), dtype=torch.float)
    ys = torch.as_tensor(np.stack(ys), dtype=torch.float)
    return xs, ys, infos

# creates datasets and dataloaders
def create_loaders(datasets, args, split_test=True, test_size=None, shuffle=True, order_fn=None):
    train_sets = []
    test_sets = []
    for d in range(len(datasets)):
        dset = load_rb(datasets[d])
        if not shuffle and order_fn is not None:
            dset = sorted(dset, key=order_fn)
        if split_test:
            cutoff = round(.9 * len(dset))
            train_sets.append(dset[:cutoff])
            test_sets.append(dset[cutoff:])
        else:
            test_sets.append(dset)

    test_set = TrialDataset(test_sets, args)
    if test_size is None:
        test_size = 128
    if split_test:
        train_set = TrialDataset(train_sets, args)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collater, shuffle=shuffle, drop_last=True)
        test_size = min(test_size, len(test_set))
        test_loader = DataLoader(test_set, batch_size=test_size, collate_fn=collater, shuffle=shuffle)
        return (train_set, train_loader), (test_set, test_loader)
    else:
        test_size = min(test_size, len(test_set))
        test_loader = DataLoader(test_set, batch_size=test_size, collate_fn=collater, shuffle=shuffle)
        return (test_set, test_loader)

def get_criteria(args):
    criteria = []
    if 'mse' in args.loss:
        fn = nn.MSELoss(reduction='sum')
        def mse(o, t, i=None):
            return args.l1 * fn(t, o)
        criteria.append(mse)
    if 'bce' in args.loss:
        weights = args.l3 * torch.ones(1)
        fn = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=weights)
        def bce(o, t, i=None):
            return args.l1 * fn(t, o)
        criteria.append(bce)
    if 'mse-w' in args.loss:
        fn = nn.MSELoss(reduction='sum')
        def mse_w(o, t, i):
            loss = 0.
            if len(o.shape) == 1:
                o = o.unsqueeze(0)
                t = t.unsqueeze(0)
                i = [i]
            for j in range(len(t)):
                t_set, t_go = i[j][1], i[j][2]
                t_p = t_go - t_set
                # using interval from t_set to t_go + t_p
                loss += t.shape[1] / t_p * fn(o[j,t_set:t_go+t_p+1], t[j,t_set:t_go+t_p+1])
            return args.l2 * loss
        criteria.append(mse_w)
    if 'bce-w' in args.loss:
        weights = args.l4 * torch.ones(1)
        fn = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=weights)
        def bce_w(o, t, i):
            loss = 0.
            if len(o.shape) == 1:
                o = o.unsqueeze(0)
                t = t.unsqueeze(0)
                i = [i]
            for j in range(len(t)):
                t_set, t_go = i[j][1], i[j][2]
                t_p = t_go - t_set
                # using interval from t_set to t_go + t_p
                # normalizing by length of the whole trial over length of penalized window
                loss += t.shape[1] / t_p * fn(o[j,t_set:t_set+t_p+1], t[j,t_set:t_go+t_p+1])
            return args.l2 * loss
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