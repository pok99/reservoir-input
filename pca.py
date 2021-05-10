import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors

import argparse
import sys
import os
import pdb

from testers import get_states, load_model_path
from helpers import create_loaders
from utils import get_config

from tasks import *

cspaces = [cm.spring, cm.summer, cm.autumn, cm.winter]

def main(args):
    config = get_config(args.model, to_bunch=True)
    net = load_model_path(args.model, config)

    if len(args.dataset) == 0:
        args.dataset = config.dataset

    n_reps = 50
    _, loader = create_loaders(args.dataset, config, split_test=False, test_size=n_reps)
    x, y, trials = next(iter(loader))
    A = get_states(net, x)

    t_type = type(trials[0])
    if t_type == RSG:
        pca_rsg(args, A, trials, n_reps)
    elif t_type in [DelayProAnti, MemoryProAnti]:
        pca_dmpa(args, A, trials, n_reps)

    # A_proj = pca(A, 3)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.grid(False)
    # plt.axis('off')

    # colors = cm.autumn(np.linspace(0, 1, 6))

    # for ix in range(A_proj.shape[0]):
    #     t = A_proj[ix].T
    #     # trial = samples['0_delaypro'][2][ix]
    #     # trial = samples['0_memorypro'][2][ix]
    #     # trial = samples['0_flip-flop-2-0:01'][2][ix]
    #     # trial = samples['0_flip-flop-1-0:04'][2][ix]
    #     # trial = samples['0_durdisc'][2][ix]

    #     ax.plot(t[0], t[1], t[2], color=colors[0], lw=1)

    #     # direction = (trial.s1[1] < trial.s2[1]) ^ (trial.cue_id == 1)
    #     # ax.scatter(t[0][0], t[1][0], t[2][0], color='salmon', marker='o')
    #     # if direction == 1:
    #     #     ax.plot(t[0], t[1], t[2], color=colors[0], lw=1)
    #     # else:
    #     #     ax.plot(t[0], t[1], t[2], color=colors[2], lw=1)


    #     # stim = trial.stim
    #     # fix = trial.fix
    #     # mem = trial.memory

    #     # ax.plot(t[0][:fix], t[1][:fix], t[2][:fix], color=colors[0], lw=1.5)
    #     # ax.plot(t[0][fix:stim], t[1][fix:stim], t[2][fix:stim], color=colors[1], lw=1.5)
    #     # ax.plot(t[0][stim:], t[1][stim:], t[2][stim:], color=colors[1], lw=1.5)
    #     # # ax.plot(t[0][stim:mem], t[1][stim:mem], t[2][stim:mem], color=colors[2], lw=1.5)
    #     # # ax.plot(t[0][mem:], t[1][mem:], t[2][mem:], color=colors[3], lw=1.5)
    #     # ax.scatter(t[0][fix], t[1][fix], t[2][fix], color='salmon', marker='s')
    #     # ax.scatter(t[0][stim], t[1][stim], t[2][stim], color='salmon', marker='^')
    #     # ax.scatter(t[0][-1], t[1][-1], t[2][-1], s=10, color='salmon', marker='o')

    # plt.show()

def pca_rsg(args, A_uncut, trials, n_reps):

    setting = 'estimation'

    As = []
    for idx in range(n_reps):
        t_ready, t_set, t_go = trials[idx].rsg
        if setting == 'estimation':
            As.append(A_uncut[idx,t_ready:t_set])
        elif setting == 'prediction':
            As.append(A_uncut[idx,t_set:t_go])

    A_proj = pca(As, 3)

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

def pca_dmpa(args, A_uncut, trials, n_reps):

    setting = 'preparation'

    As = []
    for idx in range(n_reps):
        t_type = type(trials[idx])
        fix = trials[idx].fix
        stim = trials[idx].stim    
        if t_type is MemoryProAnti:
            memory = trials[idx].memory
        if setting == 'preparation':
            if t_type is DelayProAnti:
                As.append(A_uncut[idx,fix:stim])
            else:
                As.append(A_uncut[idx,fix:memory])
        elif setting == 'movement':
            if t_type is DelayProAnti:
                As.append(A_uncut[stim:])
            else:
                As.append(A_uncut[memory:])

    A_proj = pca(As, 3)

    n_contexts = len(args.dataset)
    stimuli_groups = [{} for i in range(n_contexts)]
    for idx in range(n_reps):
        stimulus = tuple(trials[idx].stimulus)
        context = trials[idx].context
        if stimulus in stimuli_groups[context]:
            stimuli_groups[context][stimulus].append(A_proj[idx])
        else:
            stimuli_groups[context][stimulus] = [A_proj[idx]]

    context_colors = [
        iter(cspaces[i](np.linspace(0, 1, len(stimuli_groups[i])))) for i in range(n_contexts)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    plt.axis('off')

    for context, groups in enumerate(stimuli_groups):
        sorted_stimuli = sorted(groups.keys())
        for stimulus in sorted_stimuli:
            v = groups[stimulus]
            proj = sum(v) / len(v)
            c = next(context_colors[context])

            t = proj.T

            ax.plot(t[0], t[1], t[2], color=c, lw=1)
            if setting == 'preparation':
                marker_a = '^'
                marker_b = 'o'
            else:
                marker_a = 'o'
                marker_b = 's'
            ax.scatter(t[0][0], t[1][0], t[2][0], s=40, color=c, marker=marker_a)
            ax.scatter(t[0][-1], t[1][-1], t[2][-1], s=30, color=c, marker=marker_b)

    plt.show()


# As should be either [T, D] or [[T, D], ...] shaped where
# outer (optional) listing
# T is timesteps
# D is the dimensional space that needs to be reduced

def pca(As, rank):
    # can deal with either list of inputs or a single A vector
    if type(As) is not list:
        As = [As]

    N = len(As)
    # mix up the samples and timesteps, but keep the dimensions
    A = torch.cat(As)

    u, s, v = torch.pca_lowrank(A)

    projs = []
    for ix in range(N):
        traj = As[ix]
        traj_proj = traj @ v[:, :rank]
        projs.append(traj_proj)

    return projs

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model', type=str)
    ap.add_argument('-d', '--dataset', type=str, nargs='+', default=[])
    args = ap.parse_args()

    main(args)