
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler

import pdb

import random
from collections import OrderedDict

from utils import load_rb

from tasks import *

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

class TaskSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=True, generator=None):
        super().__init__(data_source)
        assert type(data_source) is TrialDataset
        max_idxs = data_source.max_idxs
        self.bounds = [[max_idxs[i-1], max_idxs[i]] for i in range(len(max_idxs))]
        self.bounds[0][0] = 0

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self):
        ranges = [b[0] + torch.randperm(b[1] - b[0], generator=self.generator) for b in self.bounds]
        batches = []
        for r in ranges:
            if self.drop_last:
                i_set = range(len(r) // self.batch_size)
            else:
                i_set = range(int(np.ceil(len(r) / self.batch_size)))
            batches += [r[self.batch_size*i:self.batch_size*(i+1)] for i in i_set]
        batches = [batches[i].tolist() for i in torch.randperm(len(batches), generator=self.generator)]
        return iter(batches)

    def __len__(self):
        lens = [(b[1] - b[0] // self.batch_size) for b in self.bounds]
        return sum(lens)


# dataset that automatically creates trials composed of trial and context data
class TrialDataset(Dataset):
    def __init__(self, datasets, args):
        self.args = args
        
        self.names = []
        self.datasets = []
        self.t_types = []
        self.lzs = []
        self.max_idxs = np.zeros(len(datasets), dtype=int)
        for i, (dname, ds) in enumerate(datasets):
            self.names.append(dname)
            self.datasets.append(ds)
            # cumulative lengths of datasets, for indexing
            self.max_idxs[i] = self.max_idxs[i-1] + len(ds)
            # use ds[0] as exemplar to set t_type, L, Z
            self.t_types.append(ds[0].t_type)
            # change Ls and Zs as they may vary for dataset subtypes
            L, Z = ds[0].L, ds[0].Z
            if args is not None and args.separate_signal and type(ds[0]) is RSG:
                L += 1
            self.lzs.append((L, Z))

    def __len__(self):
        return self.max_idxs[-1]

    def __getitem__(self, idx):
        # index into the appropriate dataset to get the trial
        context = self.get_context(idx)
        if context != 0:
            idx = idx - self.max_idxs[context-1]

        trial = self.datasets[context][idx]
        x = trial.get_x(self.args)
        y = trial.get_y(self.args)

        trial.name = self.names[context]
        trial.lz = self.lzs[context]
        trial.context = context
        trial.idx = idx
        return x, y, trial

    def get_context(self, idx):
        return np.argmax(self.max_idxs > idx)


# turns data samples into stuff that can be run through network
def collater(samples):
    xs, ys, trials = list(zip(*samples))
    xs = torch.as_tensor(np.stack(xs), dtype=torch.float)
    ys = torch.as_tensor(np.stack(ys), dtype=torch.float)
    return xs, ys, trials

# for test loaders that produce 1 sample at a time, combine them based on name of dset
def get_test_samples(loader, n_tests):
    samples = {}
    for i in range(n_tests):
        sample = next(iter(loader))[0]
        t_type = sample[2].name
        if t_type in samples:
            samples[t_type].append(sample)
        else:
            samples[t_type] = [sample]
    for t, s in samples.items():
        samples[t] = collater(s)
    ordered_samples = OrderedDict((k, samples[k]) for k in sorted(samples.keys()))
    return ordered_samples

# creates datasets and dataloaders
def create_loaders(datasets, args, split_test=True, shuffle=True, order_fn=None):
    dsets_train = []
    dsets_test = []
    for i, dpath in enumerate(datasets):
        dset = load_rb(dpath)
        # trim and set name of each dataset
        dname = str(i) + '_' + ':'.join(dpath.split('/')[-1].split('.')[:-1])
        if not shuffle and order_fn is not None:
            dset = sorted(dset, key=order_fn)
        if split_test:
            cutoff = round(.9 * len(dset))
            dsets_train.append([dname, dset[:cutoff]])
            dsets_test.append([dname, dset[cutoff:]])
        else:
            dsets_test.append([dname, dset])

    # test loader used regardless of setting
    test_set = TrialDataset(dsets_test, args)
    # sample 1 at a time, combine and collate them later on
    test_sampler = TaskSampler(test_set, 1, drop_last=False)
    test_loader = DataLoader(test_set, batch_sampler=test_sampler, collate_fn=lambda x:x)

    if split_test:
        train_set = TrialDataset(dsets_train, args)
        train_sampler = TaskSampler(train_set, args.batch_size, drop_last=True)
        train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=collater)
        return (train_set, train_loader), (test_set, test_loader)
    else:
        return (test_set, test_loader)

def get_criteria(args):
    criteria = []
    if 'mse' in args.loss:
        # do this in a roundabout way due to truncated bptt
        fn = nn.MSELoss(reduction='sum')
        def mse(o, t, i, t_ix, single=False):
            # last dimension is number of timesteps
            # divide by batch size to avoid doing so logging and in test
            # needs all the contexts to be the same length
            loss = 0.
            if single:
                o = o.unsqueeze(0)
                t = t.unsqueeze(0)
                i = [i]
            for j in range(len(t)):
                length = i[j].t_len
                if t_ix + t.shape[-1] <= length:
                    loss += fn(t[j], o[j])# / length
                elif t_ix < length:
                    t_adj = t[j,:,:length-t_ix]
                    o_adj = o[j,:,:length-t_ix]
                    loss += fn(t_adj, o_adj)# / length
            return args.l1 * loss / args.batch_size
        criteria.append(mse)
    if 'bce' in args.loss:
        weights = args.l3 * torch.ones(1)
        fn = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=weights)
        def bce(o, t, **kwargs):
            return args.l1 * fn(t, o)
        criteria.append(bce)
    if 'mse-e' in args.loss:
        # ONLY FOR RSG AND CSG, WITH [1D] OUTPUT
        # exponential decaying loss from the go time on both sides
        # loss is 1 at go time, 0.5 at set time
        # normalized to the number of timesteps taken
        fn = nn.MSELoss(reduction='none')
        def mse_e(o, t, i, t_ix, single=False):
            loss = 0.
            if single:
                o = o.unsqueeze(0)
                t = t.unsqueeze(0)
                i = [i]
            for j in range(len(t)):
                # last dimension is number of timesteps
                t_len = t.shape[-1]
                xr = torch.arange(t_len, dtype=torch.float)
                # placement of go signal in subset of timesteps
                t_g = i[j].rsg[2] - t_ix
                t_p = i[j].t_p
                # exponential loss centred at go time
                # dropping to 0.5 at set time
                lam = -np.log(2) / t_p
                # left half, only use if go time is to the right
                if t_g > 0:
                    xr[:t_g] = torch.exp(-lam * (xr[:t_g] - t_g))
                # right half, can use regardless because python indexing is nice
                xr[t_g:] = torch.exp(lam * (xr[t_g:] - t_g))
                # normalize, just numerically calculate area
                xr = xr / torch.sum(xr) * t_len
                # only the first dimension matters for rsg and csg output
                loss += torch.dot(xr, fn(o[j][0], t[j][0]))
            return args.l2 * loss / args.batch_size
        criteria.append(mse_e)
    if len(criteria) == 0:
        raise NotImplementedError
    return criteria

def get_activation(name):
    if name == 'exp':
        fn = torch.exp
    elif name == 'relu':
        fn = nn.ReLU()
    elif name == 'sigmoid':
        fn = nn.Sigmoid()
    elif name == 'tanh':
        fn = nn.Tanh()
    elif name == 'none':
        fn = lambda x: x
    return fn

def get_output_activation(args):
    return get_activation(args.out_act)

def get_dim(a):
    if hasattr(a, '__iter__'):
        return len(a)
    else:
        return 1

    return l2 * total_loss
