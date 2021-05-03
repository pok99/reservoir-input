import numpy as np
import torch
import torch.nn as nn

import random
import os
import pdb
import json
import sys

from network import BasicNetwork, M2Net
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
def test_model(net, config, n_tests=0):
    test_set, test_loader = create_loaders(config.dataset, config, split_test=False, test_size=n_tests)
    x, y, info = next(iter(test_loader))
    dset_idx = [t.n for t in info]

    # pdb.set_trace()

    criteria = get_criteria(config)

    with torch.no_grad():
        net.reset()

        # saving each individual loss per sample, per timestep
        losses = np.zeros(len(x))
        outs = []

        for j in range(x.shape[2]):
            # run the step
            net_in = x[:,:,j].reshape(-1, net.args.L + net.args.T)
            net_out = net(name, net_in)
            outs.append(net_out)

        # pdb.set_trace()
        net_outs = torch.stack(outs, dim=2)
        net_targets = y
        # pdb.set_trace()
        for c in criteria:
            for k in range(len(x)):
                losses[k] += c(net_outs[k], net_targets[k], i=info[k], t_ix=0, single=True).item()

    ins = x
    goals = y

    # if 'bce' in config.losses or 'bce-w' in config.losses:
    #     outs = torch.sigmoid(net_outs)
    # else:
    outs = net_outs

    data = list(zip(dset_idx, ins, goals, outs, losses, info))
    final_loss = np.mean(losses)
    return data, final_loss


