import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import random
import pickle
import argparse
import pdb

from dataset import load_dataset
from helpers import test_model
from reservoir import Network, Reservoir

# parser = argparse.ArgumentParser()
# parser.add_argument('model')
# parser.add_argument('dataset')
# args = parser.parse_args()

# with open(args.model, 'rb') as f:
#     model = torch.load(f)

# dset = load_dataset(args.dataset)
# data = test_model(model, dset, n_tests=12)

# run_id = '/'.join(args.file.split('/')[-4:-2])

parser = argparse.ArgumentParser()
parser.add_argument('data')
args = parser.parse_args()

with open(args.data, 'rb') as f:
    data_pre = pickle.load(f)
    dset_idx = sorted(random.sample(range(len(data_pre)), 12))
    data = [data_pre[ix] for ix in dset_idx]

    run_id = 0

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
fig.suptitle(f'Final performance: {run_id}')
fig.legend(handles, labels, loc='center right')

plt.show()


