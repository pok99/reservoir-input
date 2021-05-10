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

cols = ['cornflowerblue']

def main(args):
    config = get_config(args.model, to_bunch=True)
    net = load_model_path(args.model, config)

    if len(args.dataset) == 0:
        args.dataset = config.dataset

    n_reps = 10
    _, loader = create_loaders(args.dataset, config, split_test=False, test_size=n_reps)
    x, y, trials = next(iter(loader))
    A = get_states(net, x)

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

        # direction = (trial.s1[1] < trial.s2[1]) ^ (trial.cue_id == 1)
        # ax.scatter(t[0][0], t[1][0], t[2][0], color='salmon', marker='o')
        # if direction == 1:
        #     ax.plot(t[0], t[1], t[2], color=colors[0], lw=1)
        # else:
        #     ax.plot(t[0], t[1], t[2], color=colors[2], lw=1)


        # stim = trial.stim
        # fix = trial.fix
        # mem = trial.memory

        # ax.plot(t[0][:fix], t[1][:fix], t[2][:fix], color=colors[0], lw=1.5)
        # ax.plot(t[0][fix:stim], t[1][fix:stim], t[2][fix:stim], color=colors[1], lw=1.5)
        # ax.plot(t[0][stim:], t[1][stim:], t[2][stim:], color=colors[1], lw=1.5)
        # # ax.plot(t[0][stim:mem], t[1][stim:mem], t[2][stim:mem], color=colors[2], lw=1.5)
        # # ax.plot(t[0][mem:], t[1][mem:], t[2][mem:], color=colors[3], lw=1.5)
        # ax.scatter(t[0][fix], t[1][fix], t[2][fix], color='salmon', marker='s')
        # ax.scatter(t[0][stim], t[1][stim], t[2][stim], color='salmon', marker='^')
        # ax.scatter(t[0][-1], t[1][-1], t[2][-1], s=10, color='salmon', marker='o')

    plt.show()

# A should be [N, T, D] shaped where
# N is the number of samples
# T is timesteps
# D is the dimensional space that needs to be reduced

def pca(As, rank):
    # mix up the samples and timesteps, but keep the dimensions
    N = len(As)
    A_cut = torch.cat(As)
    u, s, v = torch.pca_lowrank(A)

    # if rank == 3:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.grid(False)
    #     plt.axis('off')

    projs = []
    for ix in range(N):
        traj = A[ix]
        traj_proj = traj @ v[:, :rank]
        projs.append(traj_proj)

    projs = torch.stack(projs, dim=0)
    return projs

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model', type=str)
    ap.add_argument('-d', '--dataset', type=str, nargs='+', default=[])
    args = ap.parse_args()

    main(args)