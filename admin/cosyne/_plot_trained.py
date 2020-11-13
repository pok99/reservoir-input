import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

import random
import pickle
import argparse
import pdb
import json

from helpers import sigmoid
from utils import load_rb, get_config, fill_undefined_args
from testers import load_model_path, test_model

# for plotting some instances of a trained model on a specified dataset

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to a model file, to be loaded into pytorch')
parser.add_argument('-d', '--dataset', help='path to a dataset of trials')
parser.add_argument('--noise', default=0, help='noise to add to trained weights')
parser.add_argument('-r', '--res_noise', default=None, type=float)
parser.add_argument('-m', '--m_noise', default=None, type=float)
parser.add_argument('-x', '--reservoir_x_init', default=None, type=str)
parser.add_argument('-a', '--test_all', action='store_true')
parser.add_argument('-n', '--no_plot', action='store_true')
parser.add_argument('-c', '--config', default=None, help='path to config file if custom')
args = parser.parse_args()

if args.config is None:
    config = get_config(args.model, ctype='model')
else:
    config = json.load(open(args.config, 'r'))
config = fill_undefined_args(args, config, overwrite_none=True)

net = load_model_path(args.model, config=config)
# assuming config is in the same folder as the model

if args.test_all:
    _, loss = test_model(net, config)
    print('avg summed loss (all):', loss)

if not args.no_plot:
    data, loss = test_model(net, config, n_tests=2)
    print('avg summed loss (plotted):', loss)
    run_id = '/'.join(args.model.split('/')[-3:-1])

    fig, ax = plt.subplots(2,1,figsize=(4, 6), squeeze=False)
    # gs1 = gridspec.GridSpec(2,1, figure=fig)
    # gs1.update(left=0.05, wspace=2, hspace=2) # set the spacing between axes. 
    plt.subplots_adjust(wspace=1, hspace=.1)
    for i, ax in enumerate(fig.axes):
        ix, x, y, z, loss = data[i]
        xr = np.arange(len(x))

        ax.axvline(x=0, color='black', alpha = 1, lw=1)
        ax.axhline(y=0, color='black', alpha = 1, lw=1)
        # ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
        ax.grid(None)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # quick fix to 2d
        if len(x.shape) > 1:
            if 'rsg' in config.dataset:
                x = x[:,0] + x[:,1]
            elif 'copy' in config.dataset:
                x = x[:,0]
        if 'rsg' in config.dataset:
            ax.scatter(xr, x, color='coral', alpha=0.5, s=3, label='input')
            ax.set_ylim([-.5,2])
        elif 'copy-snip' in config.dataset:
            ax.plot(xr[:len(xr)//2], x[:len(xr)//2], color='coral', alpha=1, lw=2, label='stimulus      ')
            ax.plot(xr, y, color='black', alpha=1, lw=1, ls='--', label='target', zorder=100)
            ax.plot(xr[len(xr)//2:], x[len(xr)//2:], color='coral', alpha=1, lw=2)
        elif 'copy' in config.dataset:
            ax.plot(xr, x, color='coral', alpha=0.5, lw=2, label='stimulus')

        if 'mse' in config.losses or 'mse-w2' in config.losses:
            ax.plot(xr, z, color='cornflowerblue', alpha=1, lw=2, label='output       ')

        # ax.tick_params(axis='both', color='white')
        ax.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        
        ax.set_xlim([-1, 200])
        ax.set_ylim([min(min(x),min(z))-.5,max(max(x),max(z))+.5])
        
        # ax.set_title(f'trial {ix}, avg loss {np.round(float(loss), 2)}', size='small')
        # ax.annotate('$t$', (195,-.1), size='large')


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    plt.show()


