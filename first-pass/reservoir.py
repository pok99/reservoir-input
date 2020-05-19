import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import pdb


# reservoir network. shouldn't be trained
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
        self.b = nn.Parameter(torch.zeros(args.D, 1))

        self.f = lambda x: self.W_f @ x + self.b

        self.W_ro = nn.Parameter(torch.randn(1, args.N))


    def forward(self, o):
        u = self.f(o.reshape(-1, 1))
        x = self.reservoir(u)
        z = self.W_ro @ x
        return z, u, x

    def reset(self):
        self.reservoir.reset()
