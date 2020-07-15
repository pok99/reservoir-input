import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import pdb

from utils import Bunch, load_rb


# reservoir network. shouldn't be trained
class Reservoir(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.J = nn.Linear(args.N, args.N, bias=False)
        self.W_u = nn.Linear(args.D, args.N, bias=False)
        self.activation = torch.tanh
        self.tau_x = 10

        self.n_burn_in = args.reservoir_burn_steps
        self.reservoir_x_seed = args.reservoir_x_seed

        if hasattr(args, 'reservoir_noise'):
            self.noise_std = args.reservoir_noise
        else:
            self.noise_std = 0

        if hasattr(args, 'reservoir_path') and args.reservoir_path is not None:
            J, W_u = load_rb(args.reservoir_path)
            self.J.weight.data = J
            self.W_u.weight.data = W_u
        else:
            self._init_J(args.res_init_type, args.res_init_params)

        self.reset()

    def _init_J(self, init_type, init_params):
        if init_type == 'gaussian':
            rng_pt = torch.get_rng_state()
            torch.manual_seed(self.args.reservoir_seed)
            self.J.weight.data = torch.normal(0, init_params['std'], self.J.weight.shape) / np.sqrt(self.args.N)
            self.W_u.weight.data = torch.normal(0, init_params['std'], self.W_u.weight.shape) / np.sqrt(self.args.N)
            torch.set_rng_state(rng_pt)

    def burn_in(self, steps):
        for i in range(steps):
            g = self.activation(self.J(self.x))
            delta_x = (-self.x + g) / self.tau_x
            self.x = self.x + delta_x
        self.x.detach_()

    def forward(self, u):
        g = self.activation(self.J(self.x) + self.W_u(u))
        if self.noise_std > 0:
            gn = g + torch.normal(torch.zeros_like(g), self.noise_std)
        else:
            gn = g
        delta_x = (-self.x + gn) / self.tau_x
        self.x = self.x + delta_x
        return self.x

    def reset(self, res_state_seed=None):
        if res_state_seed is None:
            res_state_seed = self.reservoir_x_seed
        # reset to 0 if x seed is -1
        if res_state_seed == -1:
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

        self.W_f = nn.Linear(args.L, args.D, bias=True)
        self.W_ro = nn.Linear(args.N, args.Z, bias=True)

        if hasattr(args, 'Wf_path') and args.Wf_path is not None:
            W_f = load_rb(args.Wf_path)
            # helps solve bias = False problems
            if hasattr(type(W_f), '__iter__'):
                self.W_f.weight.data = W_f[0]
                self.W_f.bias.data = W_f[1]
            self.W_f.weight.data = W_f
            #self.W_f.bias.data = self.W_f.bias * 0
        if hasattr(args, 'Wro_path') and args.Wro_path is not None:
            W_ro = load_rb(args.Wro_path)
            if hasattr(type(W_ro), '__iter__'):
                self.W_ro.weight.data = W_ro[0]
                self.W_ro.bias.data = W_ro[1]
            self.W_ro.weight.data = W_ro
            #self.W_ro.bias.data = self.W_ro.bias * 0

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
            return z2, x, u


    def reset(self, res_state_seed=None):
        self.reservoir.reset(res_state_seed=res_state_seed)
        # set up network delay mechanism. essentially a queue of length network_delay
        # with a pointer to the current index
        if self.network_delay != 0:
            self.delay_output = [None] * self.network_delay
            self.delay_ind = 0

