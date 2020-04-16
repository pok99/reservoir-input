import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import pdb

import argparse

from utils import *

def create_dataset(n_trials = 1000):
    trials = []
    trial_len = 300
    for n in range(n_trials):
        trial_x = np.zeros((trial_len))
        set_time = np.random.randint(1, trial_len / 2)
        trial_x[set_time] = 1

        trial_y = np.zeros((trial_len))
        trial_y[set_time * 2] = 1
        trials.append((trial_x, trial_y))

    return trials

def save_dataset(dset, fname):
    with open(os.path.join('data', fname), 'wb') as f:
        pickle.dump(dset, f)

def load_dataset(fpath):
    with open(fpath, 'rb') as f:
        dset = pickle.load(f)
    return dset


class Reservoir(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.J = nn.Parameter(torch.empty((args.N, args.N)))
        self.W_u = nn.Parameter(torch.empty((args.N, args.D)))
        self.activation = torch.tanh

        self._init_J(args.res_init_type, args.res_init_params)

        self.x = torch.zeros((args.N, 1))

        self.tau_x = 10

        self.args = args

    def _init_J(self, init_type, init_params):
        if init_type == 'gaussian':
            self.J.data = torch.normal(0, init_params['std'], self.J.shape)
            self.W_u.data = torch.normal(0, init_params['std'], self.W_u.shape)

    def forward(self, u):
        g = self.activation(self.J @ self.x + self.W_u @ u)
        delta_x = (-self.x + g) / self.tau_x
        self.x = self.x + delta_x
        return self.x

    def reset(self):
        self.x = torch.zeros((self.args.N, 1))

class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.reservoir = Reservoir(args)

        self.W_f = nn.Parameter(torch.randn(args.D, args.O))

        self.f = lambda x: self.W_f @ x

        self.W_ro = nn.Parameter(torch.randn(2, args.N))


    def forward(self, o):
        u = self.f(o.reshape(-1, 1))
        x = self.reservoir(u)
        z = self.W_ro @ x
        return z

    def reset(self):
        self.reservoir.reset()

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-N', default=10, help='')
    parser.add_argument('-D', default=2, help='')
    parser.add_argument('--res_init_type', default='gaussian', help='')
    parser.add_argument('--res_init_gaussian_std', default=1.5)

    # todo: arguments for res init parameters

    parser.add_argument('-O', default=1, help='')


    args = parser.parse_args()
    args.res_init_params = {}
    if args.res_init_type == 'gaussian':
        args.res_init_params['std'] = args.res_init_gaussian_std
    return args

def main():
    args = parse_args()

    dset = load_dataset('data/rsg_test.pkl')

    torch.autograd.set_detect_anomaly(True)

    net = Network(args)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.1)

    ix = 0
    for trial in dset:
        ix += 1
        x, y = torch.from_numpy(trial[0]).float(), torch.from_numpy(trial[1]).long()

        net.reset()
        optimizer.zero_grad()

        outs = []
        total_loss = torch.tensor(0.)
        for j in range(x.shape[0]):
            net_in = x[j].unsqueeze(-1)
            net_out = net(net_in)
            net_target = y[j].unsqueeze(0)
            outs.append(net_out)
            step_loss = criterion(net_out.transpose(0,1), net_target)
            total_loss += step_loss

        total_loss.backward()
        optimizer.step()
        print(total_loss)






        if ix > 60:
            break

    print(torch.stack(outs).squeeze().max(1)[1].flatten())

    pdb.set_trace()






if __name__ == '__main__':
    
    main()

    # dset = create_dataset()
    # save_dataset(dset, 'rsg_test.pkl')


