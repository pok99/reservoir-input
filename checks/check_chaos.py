import numpy as np
import torch
import matplotlib.pyplot as plt

import pdb
import sys
import random

sys.path.append('../')

from network import BasicNetwork, M2Reservoir
from utils import Bunch


colors = ['moccasin', 'springgreen', 'royalblue', 'cornflowerblue']

# plot many repetitions of a single value for g
# these are the differences between 
def single_std():
    g = 1.5
    N = 500
    r_noise = 0.1
    n_unique_nets = 9
    n_steps = 2000
    n_reps = 3
    all_dists = []
    for rep in range(n_unique_nets):
        b = Bunch(N=N, res_init_g=g, bias=False)
        net = M2Reservoir(b)
        init_x = np.random.normal(0, 1, (1, N))
        net.reset(res_state=init_x)
        xs = np.zeros((n_steps, N))
        for i in range(n_steps):
            net()
            xs[i] = net.x.detach().numpy().reshape(-1)

        dists = []

        # changing the initial condition just a lil bit
        for i in range(n_reps):
            new_x = init_x + np.random.normal(0, r_noise, (1, N))
            net.reset(res_state=new_x)
            xss = np.zeros((n_steps, N))
            dist = np.zeros(n_steps)
            for j in range(n_steps):
                net()
                xss[j] = net.x.detach().numpy().reshape(-1)
                dist[j] = np.linalg.norm(xss[j] - xs[j])

            dists.append(dist)

        all_dists.append(dists)

    plt.figure(figsize=(15,8))
    for rep in range(n_unique_nets):
        ax = plt.subplot(3, 3, rep+1)
        for j in range(n_reps):
            ax.plot(all_dists[rep][j], lw=2)

        ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
        ax.tick_params(axis='both', color='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('distance')
        ax.set_xlabel('timestep')

    plt.suptitle(f'g = {g}')
    plt.show()


if __name__ == '__main__':
    single_std()
