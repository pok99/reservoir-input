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
        self.tau_x = 10
        self.zero_reset = False

        self.n_burn_in = 400
        if hasattr(args, 'n_burn_in'):
            self.n_burn_in = args.n_burn_in

        self._init_J(args.res_init_type, args.res_init_params)
        self.reset()

    def _init_J(self, init_type, init_params):
        if init_type == 'gaussian':
            rng_pt = torch.get_rng_state()
            torch.manual_seed(self.args.reservoir_seed)
            self.J.weight.data = torch.normal(0, init_params['std'], self.J.weight.shape) / np.sqrt(self.args.N)
            self.W_u.weight.data = torch.normal(0, init_params['std'], self.W_u.weight.shape) / np.sqrt(self.args.N)
            torch.set_rng_state(rng_pt)

    def burn_in(self, steps=200):
        for i in range(steps):
            g = self.activation(self.J(self.x))
            delta_x = (-self.x + g) / self.tau_x
            self.x = self.x + delta_x
        self.x.detach_()

    def forward(self, u):
        g = self.activation(self.J(self.x) + self.W_u(u))
        delta_x = (-self.x + g) / self.tau_x
        self.x = self.x + delta_x
        return self.x

    def reset(self, res_state_seed=0):
        # don't use zero by default
        if self.zero_reset:
            self.x = torch.zeros((1, self.args.N))
        else:
            # burn in only req'd for random init because no biases to make a difference
            rng_pt = torch.get_rng_state()
            torch.manual_seed(res_state_seed)
            self.x = torch.normal(0, 1, (1, self.args.N))
            torch.set_rng_state(rng_pt)
            self.burn_in(self.n_burn_in)


class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.reservoir = Reservoir(args)

        self.W_f = nn.Linear(args.O, args.D, bias=False)
        self.W_ro = nn.Linear(args.N, 1, bias=False)

        self.network_delay = 0
        if hasattr(args, 'network_delay'):
            self.network_delay = args.network_delay

        self.reset()

    def forward(self, o):
        u = self.W_f(o.reshape(-1, 1))
        x = self.reservoir(u)
        z = self.W_ro(x)
        if self.network_delay == 0:
            return z, x, u
        else:
            z2 = self.delay_output[self.delay_ind]
            self.delay_output[self.delay_ind] = z
            self.delay_ind = (self.delay_ind + 1) % self.network_delay

            if z2 is None:
                z2 = torch.zeros_like(z)

            return z2, x, u


    def reset(self, res_state_seed=0):
        self.reservoir.reset(res_state_seed=res_state_seed)
        # set up network delay mechanism. essentially a queue of length network_delay
        # with a pointer to the current index
        if self.network_delay != 0:
            self.delay_output = [None] * self.network_delay
            self.delay_ind = 0

