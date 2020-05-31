import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import pdb

from utils import Bunch


# reservoir network. shouldn't be trained
class Reservoir(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.J = nn.Linear(args.N, args.N, bias=False)
        self.W_u = nn.Linear(args.D, args.N, bias=False)
        self.activation = torch.tanh

        self._init_J(args.res_init_type, args.res_init_params)

        self.x = torch.zeros((1, args.N))

        self.tau_x = 10

    def _init_J(self, init_type, init_params):
        if init_type == 'gaussian':
            rng_pt = torch.get_rng_state()
            torch.manual_seed(self.args.reservoir_seed)
            self.J.weight.data = torch.normal(0, init_params['std'], self.J.weight.shape) / np.sqrt(self.args.N)
            self.W_u.weight.data = torch.normal(0, init_params['std'], self.W_u.weight.shape) / np.sqrt(self.args.N)
            torch.set_rng_state(rng_pt)

    def forward(self, u):
        g = self.activation(self.J(self.x) + self.W_u(u))
        delta_x = (-self.x + g) / self.tau_x
        self.x = self.x + delta_x
        return self.x

    def reset(self):
        self.x = torch.zeros((1, self.args.N))


class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.reservoir = Reservoir(args)

        self.W_f = nn.Linear(args.O, args.D, bias=False)
        self.W_ro = nn.Linear(args.N, 1, bias=False)

    def forward(self, o):
        u = self.W_f(o.reshape(-1, 1))
        x = self.reservoir(u)
        z = self.W_ro(x)
        return z, u, x

    def reset(self):
        self.reservoir.reset()
