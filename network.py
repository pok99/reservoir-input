import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import pdb
import random
import copy

from utils import Bunch, load_rb, fill_undefined_args
from helpers import get_output_activation

default_arglist = {
    'L': 1,
    'D': 5,
    'N': 20,
    'Z': 1,
    'res_init_type': 'gaussian',
    'res_init_params': {'std': 1.5},
    'reservoir_burn_steps': 200,
    'reservoir_noise': 0,
    'reservoir_path': None,
    'bias': False,
    'Wf_path': None,
    'Wro_path': None,
    'network_delay': 0,
    'out_act': 'exp'
}
DEFAULT_ARGS = Bunch(**default_arglist)

# reservoir network. shouldn't be trained
class Reservoir(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        self.args = copy.deepcopy(args)

        if not hasattr(args, 'reservoir_seed'):
            self.args.reservoir_seed = random.randrange(1e6)
        if not hasattr(args, 'reservoir_x_seed'):
            self.args.reservoir_x_seed = np.random.randint(1e6)

        self.J = nn.Linear(args.N, args.N, bias=False)
        self.W_u = nn.Linear(args.D, args.N, bias=False)
        self.activation = torch.tanh
        self.tau_x = 10

        self.n_burn_in = self.args.reservoir_burn_steps
        self.reservoir_x_seed = self.args.reservoir_x_seed

        self.noise_std = args.reservoir_noise

        if args.reservoir_path is not None:
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

    def reset(self, res_state_seed=None, res_state=None):
        if res_state is not None:
            # load a particular specified hidden state
            self.x = torch.from_numpy(res_state).float()
            self.burn_in(self.n_burn_in)
        else:
            # load specified hidden state from seed
            if res_state_seed is None:
                res_state_seed = self.reservoir_x_seed
            # reset to 0 if x seed is -1
            if res_state_seed == -1:
                self.x = torch.zeros((1, self.args.N))
            else:
                # if any other seed set, set the net to that seed and burn in
                rng_pt = torch.get_rng_state()
                torch.manual_seed(res_state_seed)
                self.x = torch.normal(0, 1, (1, self.args.N))
                torch.set_rng_state(rng_pt)
                self.burn_in(self.n_burn_in)


class Network(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        args = fill_undefined_args(copy.deepcopy(args), DEFAULT_ARGS)
        self.args = args
        self.reservoir = Reservoir(args)
        
        if not hasattr(args, 'bias'):
            args.bias = False

        self.W_f = nn.Linear(self.args.L, self.args.D, bias=args.bias)
        self.W_ro = nn.Linear(args.N, args.Z, bias=args.bias)

        # if hasattr(args, 'Wf_path') and args.Wf_path is not None:
        #     W_f = load_rb(args.Wf_path)
        #     # helps solve bias = False problems
        #     if hasattr(type(W_f), '__iter__'):
        #         self.W_f.weight.data = W_f[0]
        #         self.W_f.bias.data = W_f[1]
        #     self.W_f.weight.data = W_f
        # if hasattr(args, 'Wro_path') and args.Wro_path is not None:
        #     W_ro = load_rb(args.Wro_path)
        #     if hasattr(type(W_ro), '__iter__'):
        #         self.W_ro.weight.data = W_ro[0]
        #         self.W_ro.bias.data = W_ro[1]
        #     self.W_ro.weight.data = W_ro

        self.network_delay = args.network_delay

        self.reset()

    def forward(self, o):
        u = self.W_f(o.reshape(-1, 1))
        x = self.reservoir(u)
        z = self.W_ro(x)
        fn = get_output_activation(self.args)
        z = fn(z)
        #z = nn.ReLU()(z)
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

