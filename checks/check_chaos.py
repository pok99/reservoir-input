import numpy as np
import torch
import matplotlib.pyplot as plt

import pdb
import sys
import random

sys.path.append('../')

from network import BasicNetwork, Reservoir
from utils import Bunch

N = 200
res_init_std = 1.5
# b = Bunch(N=N, res_init_std=res_init_std)


colors = ['moccasin', 'springgreen', 'royalblue']


# a range of stds

# see 1.5 used multiple times
# res_init_stds = [1.5] * n_net_reps

def many_stds():
    res_init_stds = [.8, 1, 1.2, 1.4, 1.5, 1.6, 1.8, 2]
    n_net_reps = 8
    n_reps = 10
    n_steps = 600

    all_xs = []
    all_dists = []

    for rep in range(n_net_reps):
        b = Bunch(N=N, res_init_std=res_init_stds[rep], bias=False)
        net = Reservoir(b)
        # net.activation = lambda x: x
        init_x = np.random.normal(0, 1, (1, N))
        net.reset(res_state=init_x)
        xs = np.zeros((n_steps, N))
        for i in range(n_steps):
            net()
            xs[i] = net.x.detach().numpy().reshape(-1)

        dists = []

        # changing the initial condition just a lil bit
        for i in range(n_reps):
            # new_x = init_x + np.random.normal(0, .1, (1, N))
            # using a totally new initial condition
            new_x = np.random.normal(0, 1, (1, N))
            net.reset(res_state=new_x)
            xss = np.zeros((n_steps, N))
            dist = np.zeros(n_steps)
            for j in range(n_steps):
                net()
                xss[j] = net.x.detach().numpy().reshape(-1)
                # dist[j] = np.linalg.norm(xss[j] - xs[j])
                dist[j] = np.linalg.norm(xss[j])

            dists.append(dist)

        all_xs.append(np.linalg.norm(xs, axis=1))
        all_dists.append(dists)

    plt.figure(figsize=(20,10))
    for i in range(n_net_reps):
        ax = plt.subplot(2, 4, i+1)
        ax.plot(all_xs[i], lw=2, color='black')
        for j in range(n_reps):
            ax.plot(all_dists[i][j], lw=1)
            ax.set_title(f'g = {res_init_stds[i]}')
            # ax.set_xlim([0, None])
            # ax.set_ylim([0, None])

    plt.show()

# plot many repetitions of a single value for g
def single_std():
    std = 1.5
    n_net_reps = 9
    n_steps = 600
    n_reps = 10
    all_xs = []
    all_dists = []
    for rep in range(n_net_reps):
        b = Bunch(N=N, res_init_std=std, bias=False)
        net = Reservoir(b)
        # net.activation = lambda x: x
        init_x = np.random.normal(0, 1, (1, N))
        net.reset(res_state=init_x)
        xs = np.zeros((n_steps, N))
        for i in range(n_steps):
            net()
            xs[i] = net.x.detach().numpy().reshape(-1)

        dists = []

        # changing the initial condition just a lil bit
        for i in range(n_reps):
            new_x = init_x + np.random.normal(0, .1, (1, N))
            # using a totally new initial condition
            # new_x = np.random.normal(0, 1, (1, N))
            net.reset(res_state=new_x)
            xss = np.zeros((n_steps, N))
            dist = np.zeros(n_steps)
            for j in range(n_steps):
                net()
                xss[j] = net.x.detach().numpy().reshape(-1)
                dist[j] = np.linalg.norm(xss[j] - xs[j])

            dists.append(dist)

        # all_xs.append(np.linalg.norm(xs, axis=1))
        all_dists.append(dists)

    plt.figure(figsize=(15,8))
    for rep in range(n_net_reps):
        ax = plt.subplot(3, 3, rep+1)
        # ax.plot(all_xs[rep], lw=2, color='black')
        for j in range(n_reps):
            ax.plot(all_dists[rep][j], lw=1)

    plt.suptitle(f'g = {std}')
    plt.show()

# for a single value of g, plot with different amounts of noise values
def single_std_rep():
    std = 1.5
    n_net_reps = 3
    n_noises = 4
    noises = [.0001, .01, .1, 1]
    n_steps = 600
    n_reps = 10
    all_dists = []

    deltas = np.zeros((n_noises, n_reps, N))
    for i, noise in enumerate(noises):
        deltas[i, :, :] = np.random.normal(0, noise, (n_reps, N))

    init_x = np.random.normal(0, 1, (1, N))

    for rep in range(n_net_reps):
        b = Bunch(N=N, res_init_std=std, bias=False, res_seed=0)
        net = Reservoir(b)
        # net.activation = lambda x: x
        init_x = np.random.normal(0, 1, (1, N))
        net.reset(res_state=init_x)
        xs = np.zeros((n_steps, N))
        for i in range(n_steps):
            net()
            xs[i] = net.x.detach().numpy().reshape(-1)
        

        for rep_seed in range(n_noises):
            dists = []
            # changing the initial condition just a lil bit
            for i in range(n_reps):
                # new_x = init_x + np.random.normal(0, .1, (1, N))
                new_x = init_x + deltas[rep_seed, i, :]
                # using a totally new initial condition
                # new_x = np.random.normal(0, 1, (1, N))
                net.reset(res_state=new_x)
                xss = np.zeros((n_steps, N))
                dist = np.zeros(n_steps)
                for j in range(n_steps):
                    net()
                    xss[j] = net.x.detach().numpy().reshape(-1)
                    dist[j] = np.linalg.norm(xss[j] - xs[j])

                dists.append(dist)

            # all_xs.append(np.linalg.norm(xs, axis=1))
            all_dists.append(dists)

    fig = plt.figure(figsize=(15,8))
    for rep in range(n_net_reps * n_noises):
        ax = plt.subplot(n_net_reps, n_noises, rep+1)
        # ax.plot(all_xs[rep], lw=2, color='black')
        for j in range(n_reps):
            ax.plot(all_dists[rep][j], lw=1)
            ax.xlabel()

    plt.suptitle(f'g = {std}')
    fig.text(0.5, 0.04, 'noise amount', ha='center')
    fig.text(0.04, 0.5, 'initial condition', va='center', rotation='vertical')
    plt.show()

if __name__ == '__main__':
    mode = 'many_stds'
    mode = 'single_std'
    mode = 'single_std_rep'

    if mode == 'many_stds':
        many_stds()

    elif mode == 'single_std':
        single_std()

    elif mode == 'single_std_rep':
        single_std_rep()
