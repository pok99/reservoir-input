import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors

import os
import sys
import json
import pdb

import argparse


from testers import load_model_path
from utils import get_config, load_rb
from helpers import get_x_y_info



def pca(args):
    config = get_config(args.model, to_bunch=True)
    net = load_model_path(args.model, config)

    if args.dataset is None:
        args.dataset = config.dataset
    dset = load_rb(args.dataset)


    nt = 800

    goal_colors = iter(cm.cool(np.linspace(.2, 1, nt)))

    # dset = dset[:nt]
    dset.sort(key = lambda x: x[2][2]-x[2][1])

    interval = int(len(dset) / nt)
    dset = dset[::interval]

    x, y, info = get_x_y_info(config, dset)

    outs = []
    with torch.no_grad():
        net.reset()

        # saving each individual loss per sample, per timestep
        # losses = np.zeros(len(x.shape[0]))
        

        for j in range(x.shape[1]):
            # run the step
            net_in = x[:,j].reshape(-1, net.args.L)
            net_out, extras = net(net_in, extras=True)
            outs.append(extras['x'])

        # net_outs = torch.cat(outs, dim=1)
        # net_targets = y
        # for c in criteria:
        #     for k in range(len(test_set)):
        #         losses[k] += c(net_outs[k], net_targets[k], info[k]).item()

    A = torch.stack(outs, dim=1)
    A_cut = []
    for ix in range(x.shape[0]):
        A_cut.append(A[ix][info[ix][1]:info[ix][2]])
    A_cut = torch.cat(A_cut)
    u, s, v = torch.pca_lowrank(A_cut)



    # when not using categories
    # A_proj = []
    # for ix in range(x.shape[0]):
    #     traj = A[ix]
    #     tinfo = info[ix]
    #     traj_cut = traj[tinfo[0]:tinfo[1]]
    #     A_proj.append(traj_cut @ v[:, :2])

    # when using categories
    rank = 3
    A_categories = {}
    for ix in range(x.shape[0]):
        traj = A[ix]
        tinfo = info[ix]
        interval = tinfo[1] - tinfo[0]
        # traj_cut = traj[tinfo[0]:tinfo[1]]
        traj_cut = traj[tinfo[1]:tinfo[2]]
        if interval in A_categories:
            A_categories[interval].append(traj_cut @ v[:, :rank])
        else:
            A_categories[interval] = [traj_cut @ v[:, :rank]]

    goal_colors = iter(cm.cool(np.linspace(0, 1, len(A_categories))))
    
    if rank == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        plt.axis('off')

    for k,v in A_categories.items():
        proj = sum(v) / len(v)

        t = proj.T
        if rank == 2:
            plt.plot(t[0], t[1], color=next(goal_colors), lw=1)
            plt.scatter(t[0][0], t[1][0], s=20, color='black')
            plt.annotate(k, (t[0][-1], t[1][-1]))
        elif rank == 3:
            
            c = next(goal_colors)
            ax.plot(t[0], t[1], t[2], color=c, lw=1)
            # source
            ax.scatter(t[0][0], t[1][0], t[2][0], s=40, color=c, marker='^')
            # ax.annotate(k, (t[0][-1], t[1][-1], t[2][-1]))
            # ax.text(t[0][-1], t[1][-1], t[2][-1], k, color=c)
            # ending
            ax.scatter(t[0][-1], t[1][-1], t[2][-1], s=30, color=c, marker='o')




    # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)


    # for i in range(nt):
    #     t = A_proj[i].T
    #     # goal_colors = iter(cm.Oranges(np.linspace(.3, 1, t.shape[1])))
    #     # ax.scatter(t[0], t[1], t[2], color=next(goal_colors))
    #     ax.plot(t[0], t[1], color=next(goal_colors), lw=.8)
    #     ax.scatter(t[0][0], t[1][0], s=20, color='black')
    #     ax.annotate(i, (t[0][0], t[1][0]))
    #     print(len(t[0]))


    plt.show()


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('model', type=str)
    ap.add_argument('-d', '--dataset', type=str, default=None)
    args = ap.parse_args()


    pca(args)