import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors

import os
import sys
import json
import pdb

import argparse

from testers import load_model_path, get_states
from utils import get_config, load_rb
from helpers import TrialDataset, collater, create_loaders

from tasks import *

from pca import pca2

cspaces = [cm.autumn, cm.cool]

def pca(args):
    config = get_config(args.model, to_bunch=True)
    net = load_model_path(args.model, config)

    if len(args.dataset) == 0:
        args.dataset = config.dataset

    setting = 'estimation'

    n_reps = 100
    _, loader = create_loaders(args.dataset, config, split_test=False, test_size=n_reps)
    x, y, trials = next(iter(loader))
    A_uncut = get_states(net, x)

    As = []
    for idx in range(n_reps):
        t_ready, t_set, t_go = trials[idx].rsg
        if setting == 'estimation':
            As.append(A_uncut[idx,t_ready:t_set])
        elif setting == 'prediction':
            As.append(A_uncut[idx,t_set:t_go])

    A_proj = pca2(As, 3)

    n_contexts = len(args.dataset)
    interval_groups = [{} for i in range(n_contexts)]
    for idx in range(n_reps):
        rsg = trials[idx].rsg
        context = trials[idx].context
        interval = rsg[1] - rsg[0]
        if interval in interval_groups[context]:
            interval_groups[context][interval].append(A_proj[idx])
        else:
            interval_groups[context][interval] = [A_proj[idx]]

    context_colors = [
        iter(cspaces[i](np.linspace(0, 1, len(interval_groups[i])))) for i in range(n_contexts)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    plt.axis('off')

    rank = 3
    for context, groups in enumerate(interval_groups):
        sorted_intervals = sorted(groups.keys())
        for interval in sorted_intervals:
            v = groups[interval]
            proj = sum(v) / len(v)
            c = next(context_colors[context])

            t = proj.T

            ax.plot(t[0], t[1], t[2], color=c, lw=1)
            if setting == 'estimation':
                marker_a = '^'
                marker_b = 'o'
            else:
                marker_a = 'o'
                marker_b = 's'
            ax.scatter(t[0][0], t[1][0], t[2][0], s=40, color=c, marker=marker_a)
            ax.scatter(t[0][-1], t[1][-1], t[2][-1], s=30, color=c, marker=marker_b)

    plt.show()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model', type=str)
    ap.add_argument('-d', '--dataset', type=str, nargs='+', default=[])
    args = ap.parse_args()

    pca(args)