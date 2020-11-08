import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib.cm as cm

import random
import pickle
import argparse
import pdb

from helpers import sigmoid
from utils import load_rb, get_config, fill_undefined_args
from testers import load_model_path, test_model

# for plotting some instances of a trained model on a specified dataset

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to a model file, to be loaded into pytorch')
parser.add_argument('-d', '--dataset', help='path to a dataset of trials')
parser.add_argument('--noise', default=0, help='noise to add to trained weights')
parser.add_argument('--res_noise', default=None, type=float)
parser.add_argument('-x', '--reservoir_x_init', default=None, type=str)
parser.add_argument('-a', '--test_all', action='store_true')
parser.add_argument('-n', '--no_plot', action='store_true')
args = parser.parse_args()

config = get_config(args.model)
config = fill_undefined_args(args, config, overwrite_none=True)

net = load_model_path(args.model, config=config)
# dset = load_rb(args.dataset)
# assuming config is in the same folder as the model

if args.test_all:
    _, loss = test_model(net, config)
    print('avg summed loss (all):', loss)

if not args.no_plot:
    data, loss = test_model(net, config, n_tests=12)
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

        # if config.dset_type == 'rsg-gaussian':
        if config.loss == 'mse':
            ax.plot(xr, x, color='coral', alpha=0.5, lw=1, label='input')
            ax.plot(xr, y, color='coral', alpha=1, lw=1, label='target')
            ax.plot(xr, z, color='cornflowerblue', alpha=1, lw=1.5, label='response')
        elif config.loss == 'bce-pulse':
            ax.plot(xr, x, color='coral', alpha=0.5, lw=1, label='input')
            ax.plot(xr, sigmoid(z), color='cornflowerblue', alpha=1, lw=1.5, label='response')
        elif config.loss == 'mse2':
            if config.dset_type == 'rsg-pulse':
                ax.scatter(xr, x, color='coral', alpha=1, s=3, label='input')
                start = torch.nonzero(x)[0,0]
            elif config.dset_type == 'rsg-pulse2d':
                # pdb.set_trace()
                ax.scatter(xr, x[:,0], color='coral', alpha=1, s=3, label='input')
                ax.scatter(xr, x[:,1], color='coral', alpha=1, s=3, label='input')
                start = torch.nonzero(x[:,0])[0,0]
            yr = torch.nonzero(y)[0]
            ax.scatter(yr, 1, color='forestgreen', alpha=1, s=5, label='target')
            ax.plot(xr, z, color='cornflowerblue', alpha=1, lw=1.5, label='response')
            indx = torch.nonzero(z[start:] >= 1)
            if len(indx) > 0:
                indx = indx[0,0] + start
                ml, sl, bl = ax.stem([indx], [z[indx]], use_line_collection=True, linefmt='cornflowerblue', markerfmt=' ')
                sl.set_linewidth(1)

        ax.tick_params(axis='both', color='white')
        ax.set_title(f'trial {ix}, avg loss {np.round(float(loss), 2)}', size='small')
        ax.set_ylim([-.5,2])

    fig.text(0.5, 0.04, 'timestep', ha='center', va='center')
    fig.text(0.06, 0.5, 'value', ha='center', va='center', rotation='vertical')

    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(f'Final performance: {run_id}')
    fig.legend(handles, labels, loc='center right')

    plt.show()


