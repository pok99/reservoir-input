import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import random
import pickle
import argparse
import pdb

from dataset import load_dataset
from reservoir import Network, Reservoir

parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('dataset')

args = parser.parse_args()

with open(args.file, 'rb') as f:
    model = torch.load(f)


args.N = model['reservoir.J'].shape[0]
args.D = model['reservoir.W_u'].shape[1]
args.O = model['W_f'].shape[1]

args.res_init_type = 'gaussian'
args.res_init_params = {'std': 1.5}
args.reservoir_seed = 0



net = Network(args)
net.load_state_dict(model)
net.eval()

criterion = nn.MSELoss()

dset = load_dataset(args.dataset)

dset_idx = sorted(random.sample(range(len(dset)), 12))

xs = []
ys = []
zs = []
losses = []

ix = 0
for ix in range(12):
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
    print(f'Finished evaluating trial {ix}')

data = list(zip(dset_idx, xs, ys, zs, losses))

fig, ax = plt.subplots(3,4,sharex=True, sharey=True, figsize=(12,7))

for i, ax in enumerate(fig.axes):
    ix, x, y, z, loss = data[i]
    xr = np.arange(len(x))

    ax.axvline(x=0, color='dimgray', alpha = 1)
    ax.axhline(y=0, color='dimgray', alpha = 1)
    ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.plot(xr, x, color='coral', alpha=0.5, lw=1, label='ready/set')
    ax.plot(xr, y, color='coral', alpha=1, lw=1, label='go')
    ax.plot(xr, z, color='cornflowerblue', alpha=1, lw=1.5, label='response')

    ax.tick_params(axis='both', color='white')

    ax.set_title(f'trial {ix}, avg loss {round(loss, 1)}', size='small')
    ax.set_ylim([-2,2])

fig.text(0.5, 0.04, 'timestep', ha='center', va='center')
fig.text(0.06, 0.5, 'value', ha='center', va='center', rotation='vertical')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')

plt.show()
