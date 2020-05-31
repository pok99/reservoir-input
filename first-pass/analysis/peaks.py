import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import random
import pickle
import argparse
import pdb
import sys

sys.path.append('../')

from utils import Bunch

from dataset import load_dataset
from reservoir import Network, Reservoir

parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('dataset')
args = parser.parse_args()

with open(args.file, 'rb') as f:
    model = torch.load(f)

dset = load_dataset(args.dataset)

net = nn.Linear(5, 5)
pdb.set_trace()


def test_model(m_dict, dset, n_tests=0):
    bunch = Bunch()
    bunch.N = m_dict['reservoir.J'].shape[0]
    bunch.D = m_dict['reservoir.W_u'].shape[1]
    bunch.O = m_dict['W_f'].shape[1]

    bunch.res_init_type = 'gaussian'
    bunch.res_init_params = {'std': 1.5}
    bunch.reservoir_seed = 0

    net = Network(bunch)
    net.load_state_dict(m_dict)
    net.eval()

    criterion = nn.MSELoss()

    dset_idx = range(len(dset))
    if n_tests != 0:
        dset_idx = sorted(random.sample(range(len(dset)), n_tests))

    xs = []
    ys = []
    zs = []
    losses = []

    ix = 0
    for ix in range(len(dset_idx)):
        trial = dset[dset_idx[ix]]
        # next data sample
        x = torch.from_numpy(trial[0]).float()
        y = torch.from_numpy(trial[1]).float()
        xs.append(x)
        ys.append(y)

        net.reset()

        outs = []

        total_loss = torch.tensor(0.)
        for j in range(x.shape[0]):
            # run the step
            net_in = x[j].unsqueeze(0)
            net_out, val_thal, val_res = net(net_in)

            # this is the desired output
            net_target = y[j].unsqueeze(0)
            outs.append(net_out.item())

            # this is the loss from the step
            step_loss = criterion(net_out.view(1), net_target)

            total_loss += step_loss

        z = np.stack(outs).squeeze()
        zs.append(z)
        losses.append(total_loss.item())

    data = list(zip(dset_idx, xs, ys, zs, losses))

    return data

data = test_model(model, dset, 0)
pdb.set_trace()