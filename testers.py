import numpy as np
import torch
import torch.nn as nn

import random
import os
import pdb
import json
import sys

from network import M2Net
from utils import Bunch, load_rb, get_config

from helpers import get_criteria, create_loaders


def load_model_path(path, config=None):
    if config is None:
        config = get_config(path)
    if type(config) is dict:
        config = Bunch(**config)
    config.model_path = path

    # net = BasicNetwork(config)
    net = M2Net(config)

    net.eval()
    return net

# given a model and a dataset, see how well the model does on it
# works with plot_trained.py
def test_model(net, config, n_tests=128):
    if config.sequential:
        config.sequential = False
    test_set, test_loader = create_loaders(config.dataset, config, split_test=False, test_size=n_tests)
    x, y, trials = next(iter(test_loader))

    criteria = get_criteria(config)

    t_losses = {}
    with torch.no_grad():
        contexts = [t.context for t in trials]
        idxs = [t.n for t in trials]

        # saving each individual loss per sample, per timestep
        losses = np.zeros(len(x))
        outs = []

        for j in range(x.shape[2]):
            # run the step
            net_in = x[:,:,j]
            net_out = net(net_in)
            outs.append(net_out)

        outs = torch.stack(outs, dim=2)
        targets = y
        # pdb.set_trace()
        
        for k in range(len(x)):
            t = trials[k]
            for c in criteria:
                losses[k] += c(outs[k], targets[k], i=trials[k], t_ix=0, single=True).item()
            if t.dname in t_losses:
                t_losses[t.dname].append(losses[k])
            else:
                t_losses[t.dname] = [losses[k]]


        data = list(zip(contexts, idxs, trials, x, y, outs, losses))
        for name in t_losses.keys():
            t_losses[name] = np.mean(t_losses[name])

    return data, t_losses

# returns hidden states as [N, T, H]
# note: this returns hidden states as the last dimension, not timesteps!
def get_states(net, x):
    states = []
    with torch.no_grad():
        net.reset()
        for j in range(x.shape[2]):
            net_in = x[:,:,j]
            net_out, extras = net(net_in, extras=True)
            states.append(extras['x'])

    A = torch.stack(states, dim=1)
    return A