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

from helpers import get_x_y_info, mse2_loss, get_criterion


def load_model_path(path, config):
    if type(config) is dict:
        config = Bunch(**config)
    config.model_path = path

    net = BasicNetwork(config)

    net.eval()
    return net

# given a model and a dataset, see how well the model does on it
# works with plot_trained.py
def test_model(net, config, n_tests=0):
    dset = load_rb(config.dataset)
    dset_idx = range(len(dset))
    if n_tests != 0:
        dset_idx = sorted(random.sample(range(len(dset)), n_tests))
    test_set = [dset[i] for i in dset_idx]
    x, y, info = get_x_y_info(test_set)

    criterion = get_criterion(config)

    with torch.no_grad():
        net.reset()

        # saving each individual loss per sample, per timestep
        losses = np.zeros((len(test_set), x.shape[1]))
        outs = []

        for j in range(x.shape[1]):
            # run the step
            net_in = x[:,j].reshape(-1, net.args.L)
            net_out = net(net_in)
            outs.append(net_out)
            # if config.dset_type == 'rsg-gaussian':
            net_target = y[:,j].reshape(-1, net.args.Z)
            # elif config.dset_type == 'rsg-pulse':
            #     net_target = torch.zeros_like(net_out)

            for k in range(len(test_set)):
                step_loss = criterion(net_out[k], net_target[k])
                losses[k, j] = step_loss.item()

        outs = torch.cat(outs, dim=1)
        if config.loss == 'mse2':
            m_losses = np.zeros(len(test_set))
            for j in range(len(test_set)):
                m_loss = mse2_loss(x[j], outs[j], info[j], config.mse2_l1, config.mse2_l2)
                m_losses[j] = m_loss

    ins = x
    goals = y

    losses = np.sum(losses, axis=1)
    if config.loss == 'mse2':
        losses = losses + m_losses

    data = list(zip(dset_idx, ins, goals, outs, losses))

    final_loss = np.mean(losses)

    return data, final_loss


