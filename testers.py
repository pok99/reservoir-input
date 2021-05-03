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

from helpers import get_criteria, create_loaders, get_test_samples


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
    test_set, test_loader = create_loaders(config.dataset, config, split_test=False)
    samples = get_test_samples(test_loader, n_tests=n_tests)

    criteria = get_criteria(config)

    datas = []
    t_losses = []
    with torch.no_grad():
        for t, s in samples.items():
            net.reset()
            x, y, trials = s
            lz = trials[0].lz
            contexts = [t.context for t in trials]
            idxs = [t.idx for t in trials]

            # saving each individual loss per sample, per timestep
            losses = np.zeros(len(x))
            outs = []

            for j in range(x.shape[2]):
                # run the step
                net_in = x[:,:,j].reshape(-1, lz[0])
                net_out = net(t, net_in)
                outs.append(net_out)

            outs = torch.stack(outs, dim=2)
            targets = y
            # pdb.set_trace()
            for c in criteria:
                for k in range(len(x)):
                    losses[k] += c(outs[k], targets[k], i=trials[k], t_ix=0, single=True).item()


            data = list(zip(contexts, idxs, trials, x, y, outs, losses))
            datas += data
            t_losses.append((t, np.mean(losses)))
    return datas, t_losses


