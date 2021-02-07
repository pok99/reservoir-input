import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import pdb
import random
import copy
import sys

from utils import Bunch, load_rb, fill_undefined_args
from helpers import get_output_activation

DEFAULT_ARGS = {
    'L': 2,
    'D': 5,
    'N': 50,
    'Z': 2,

    'use_reservoir': True,
    'res_init_std': 1.5,
    'res_burn_steps': 200,
    'res_noise': 0,
    'bias': True,
    'network_delay': 0,
    'out_act': 'none',
    # 'stride': 1,
    'model_path': None,

    'res_path': None
}

class Reservoir(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        self.args = fill_undefined_args(args, DEFAULT_ARGS, to_bunch=True)

        if not hasattr(self.args, 'res_seed'):
            self.args.res_seed = random.randrange(1e6)
        if not hasattr(self.args, 'res_x_seed'):
            self.args.res_x_seed = np.random.randint(1e6)

        self.tau_x = 10

        self._init_vars()
        self.reset()

    def _init_vars(self):
        rng_pt = torch.get_rng_state()
        torch.manual_seed(self.args.res_seed)
        self.W_u = nn.Linear(self.args.D, self.args.N, bias=False)
        self.W_u.weight.data = torch.rand(self.W_u.weight.shape) * 2 - 1
        self.J = nn.Linear(self.args.N, self.args.N, bias=self.args.bias)
        self.J.weight.data = torch.normal(0, 1/N, self.J.weight.shape)
        self.W_ro = nn.Linear(self.args.N, self.args.Z, bias=self.args.bias)
        # print(self.J.weight.data[0])
        torch.set_rng_state(rng_pt)

        # if self.args.J_path is not None:
        #     self.load_state_dict(torch.)

        if self.args.res_path is not None:
            self.load_state_dict(torch.load(self.args.res_path))

    def burn_in(self, steps):
        for i in range(steps):
            g = torch.tanh(self.J(self.x))
            delta_x = (-self.x + g) / self.tau_x
            self.x = self.x + delta_x
        self.x.detach_()

    # extras currently doesn't do anything. maybe add x val, etc.
    def forward(self, u=None, extras=False):
        if self.nonlinear_mode == 0:

            if u is None:
                g = self.activation(self.J(self.x))
            else:
                g = self.activation(self.J(self.x) + self.W_u(u))
            # adding any inherent reservoir noise
            if self.args.res_noise > 0:
                gn = g + torch.normal(torch.zeros_like(g), self.args.res_noise)
            else:
                gn = g
            delta_x = (-self.x + gn) / self.tau_x
            self.x = self.x + delta_x

            z = self.W_ro(self.x)

        elif self.nonlinear_mode == 1:
            if u is None:
                g = self.J(self.r)
            else:
                g = self.J(self.r) + self.W_u(u)
            if self.args.res_noise > 0:
                gn = g + torch.normal(torch.zeros_like(g), self.args.res_noise)
            else:
                gn = g
            delta_x = (-self.x + gn) / self.tau_x
            self.x = self.x + delta_x
            self.r = self.activation(self.x)

            z = self.W_ro(self.r)

        if extras:
            etc = {'x': self.x.detach()}
            return z, etc
        return z

    def reset(self, res_state=None, burn_in=True):
        self.x = torch.rand(self.args.N)
        self.r = torch.tanh(self.x)

        if burn_in:
            self.burn_in(self.args.res_burn_steps)


class BasicNetwork(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        args = fill_undefined_args(args, DEFAULT_ARGS, to_bunch=True)
        self.args = args
       
        if not hasattr(self.args, 'network_seed'):
            self.args.network_seed = random.randrange(1e6)

        self._init_vars()
        if self.args.model_path is not None:
            self.load_state_dict(torch.load(self.args.model_path))
        
        self.out_act = get_output_activation(self.args)
        self.network_delay = args.network_delay

        self.reset()

    def _init_vars(self):
        rng_pt = torch.get_rng_state()
        torch.manual_seed(self.args.network_seed)
        self.W_f = nn.Linear(self.args.L, self.args.D, bias=self.args.bias)
        if self.args.use_reservoir:
            self.reservoir = Reservoir(self.args)
        else:
            self.W_ro = nn.Linear(self.args.D, self.args.Z, bias=self.args.bias)
        torch.set_rng_state(rng_pt)

    def forward(self, o, extras=False):
        # pdb.set_trace()
        # pass through the forward part
        u = self.W_f(o.reshape(-1, self.args.L))
        if self.args.use_reservoir:
            z, etc = self.reservoir(u, extras=True)
        else:
            z = self.W_ro(u)
        z = self.out_act(z)

        # if self.network_delay > 0:
        #     z_delayed = self.delay_output[self.delay_ind]
        #     self.delay_output[self.delay_ind] = z
        #     self.delay_ind = (self.delay_ind + 1) % self.network_delay
        #     z = z_delayed

        if not extras:
            return z
        elif self.args.use_reservoir:
            return z, {'x': etc['x'], 'u': u}
        else:
            return z, {'u': u}


    def reset(self, res_state=None):
        if self.args.use_reservoir:
            self.reservoir.reset(res_state=res_state)
        # set up network delay mechanism. essentially a queue of length network_delay
        # with a pointer to the current index
        # if self.network_delay > 0:
        #     self.delay_output = [None] * self.network_delay
        #     self.delay_ind = 0