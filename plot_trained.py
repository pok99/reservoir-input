import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import random
import pickle
import argparse
import pdb

from utils import load_rb
from testers import load_model_path, test_model

# for plotting some instances of a trained model on a specified dataset

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to a model file, to be loaded into pytorch')
parser.add_argument('dataset', help='path to a dataset of trials')
parser.add_argument('--noise', default=0, help='noise to add to trained weights')
parser.add_argument('--out_act', default=None, type=str)
parser.add_argument('--stride', default=1, type=int)
parser.add_argument('-a', '--test_all', action='store_true')
parser.add_argument('-n', '--no_plot', action='store_true')
args = parser.parse_args()

with open(args.model, 'rb') as f:
    model = torch.load(f)

if args.noise != 0:
    J = model['W_f.weight']
    v = J.std()
    shp = J.shape
    model['W_f.weight'] += torch.normal(0, v * .5, shp)

    J = model['W_ro.weight']
    v = J.std()
    shp = J.shape
    model['W_ro.weight'] += torch.normal(0, v * .5, shp)

net = load_model_path(args.model, params={'dset': args.dataset, 'out_act': args.out_act, 'stride': args.stride})
dset = load_rb(args.dataset)

if args.test_all:
    _, loss2 = test_model(net, dset)
    print('avg summed loss (all):', loss2)

if not args.no_plot:
    data, loss = test_model(net, dset, n_tests=12)
    print('avg summed loss (plotted):', loss)

    run_id = '/'.join(args.model.split('/')[-3:-1])

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

        ax.plot(xr, x, color='coral', alpha=0.5, lw=1, label='input')
        ax.plot(xr, y, color='coral', alpha=1, lw=1, label='target')
        ax.plot(xr, z, color='cornflowerblue', alpha=1, lw=1.5, label='response')

        ax.tick_params(axis='both', color='white')
        ax.set_title(f'trial {ix}, avg loss {np.round(float(loss), 2)}', size='small')
        ax.set_ylim([-2,3])

    fig.text(0.5, 0.04, 'timestep', ha='center', va='center')
    fig.text(0.06, 0.5, 'value', ha='center', va='center', rotation='vertical')

    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(f'Final performance: {run_id}')
    fig.legend(handles, labels, loc='center right')

    plt.show()


