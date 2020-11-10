import numpy as np
import torch
import torch.nn as nn

import random
import os
import pdb
import json
import sys

from network import BasicNetwork
from utils import Bunch, load_rb

from helpers import get_x_y_info, mse2_loss, get_criteria, shift_ix


def load_model_path(path, config):
    if type(config) is dict:
        config = Bunch(**config)
    config.model_path = path

    net = BasicNetwork(config)

    net.eval()
    return net

# given a model and a dataset, see how well the model does on it
# works with plot_trained.py
def test_model(net, config, n_tests=0, dset_base='.'):
    dset_path = os.path.join(dset_base, config.dataset)
    dset = load_rb(dset_path)
    dset_idx = range(len(dset))
    if n_tests != 0:
        dset_idx = sorted(random.sample(range(len(dset)), n_tests))
    test_set = [dset[i] for i in dset_idx]
    x, y, info = get_x_y_info(config, test_set)
    pdb.set_trace()
    x = shift_ix(config, x, info)

    criteria = get_criteria(config)

    with torch.no_grad():
        net.reset()

        # saving each individual loss per sample, per timestep
        losses = np.zeros(len(test_set))
        outs = []

        for j in range(x.shape[1]):
            # run the step
            net_in = x[:,j].reshape(-1, net.args.L)
            net_out = net(net_in)
            outs.append(net_out)

        net_outs = torch.cat(outs, dim=1)
        net_targets = y
        for c in criteria:
            for k in range(len(test_set)):
                losses[k] += c(net_outs[k], net_targets[k], info[k]).item()

    ins = x
    goals = y

    if 'bce' in config.losses or 'bce-w' in config.losses:
        outs = torch.sigmoid(net_outs)
    else:
        outs = net_outs

    data = list(zip(dset_idx, ins, goals, outs, losses))
    final_loss = np.mean(losses)
    return data, final_loss


