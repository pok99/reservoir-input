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

def pca(args):
    config = get_config(args.model, to_bunch=True)
    net = load_model_path(args.model, config)

    if len(args.dataset) == 0:
        args.dataset = config.dataset

    setting = 'estimation'

    n_reps = 10
    _, loader = create_loaders(args.dataset, config, split_test=False, test_size=n_reps)
    x, y, trials = next(iter(loader))
    A = get_states(net, x)

    pdb.set_trace()

    if setting == 'estimation':
        for state in A:
            pass

    A_proj = pca(A, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    plt.axis('off')

    colors = cm.autumn(np.linspace(0, 1, 6))

    for ix in range(A_proj.shape[0]):
        t = A_proj[ix].T
        # trial = samples['0_delaypro'][2][ix]
        # trial = samples['0_memorypro'][2][ix]
        # trial = samples['0_flip-flop-2-0:01'][2][ix]
        # trial = samples['0_flip-flop-1-0:04'][2][ix]
        # trial = samples['0_durdisc'][2][ix]

        ax.plot(t[0], t[1], t[2], color=colors[0], lw=1)


    outs = []
    with torch.no_grad():
        net.reset()
        for j in range(x.shape[2]):
            net_in = x[:,:,j].reshape(-1, net.args.L + net.args.T)
            net_out, extras = net(net_in, extras=True)
            outs.append(extras['x'])

    A = torch.stack(outs, dim=1)
    A_cut = []
    # only choosing the relevant timestamps within which to do pca
    for ix in range(x.shape[0]):
        rsg = info[ix]['rsg']
        if setting == 'estimation':
            A_cut.append(A[ix,rsg[0]:rsg[1]])
        else:
            A_cut.append(A[ix,rsg[1]:rsg[2]])
    A_cut = torch.cat(A_cut)
    u, s, v = torch.pca_lowrank(A_cut)

    rank = 3
    if rank == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        plt.axis('off')

    # when using categories WITH CONTEXTS
    context_interval_As = {}
    context_counts = np.ones(10).astype(int)
    for ix in range(x.shape[0]):
        traj = A[ix]
        rsg = info[ix]['rsg']
        context = info[ix]['context']
        interval = info[ix]['t_p']
        if setting == 'estimation':
            traj_cut = traj[rsg[0]:rsg[1]]
        else:
            traj_cut = traj[rsg[1]:rsg[2]]
        traj_proj = traj_cut @ v[:, :rank]
        group = (context, interval)
        if group in context_interval_As:
            context_interval_As[group].append(traj_proj)
        else:
            context_counts[context] += 1
            context_interval_As[group] = [traj_proj]

    context_colors = [
        iter(cm.autumn(np.linspace(0, 1, context_counts[0]))),
        iter(cm.winter(np.linspace(0, 1, context_counts[1])))
    ]

    sorted_keys = sorted(context_interval_As.keys(), key=lambda x: x[1])
    for k in sorted_keys:
        v = context_interval_As[k]
        proj = sum(v) / len(v)
        c = next(context_colors[k[0]])

        t = proj.T
        if rank == 2:
            plt.plot(t[0], t[1], color=c, lw=1)
            plt.scatter(t[0][0], t[1][0], s=20, color=c)
            plt.annotate(k, (t[0][-1], t[1][-1]))
        elif rank == 3:
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