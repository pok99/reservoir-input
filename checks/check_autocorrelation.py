import numpy as np
import torch
import matplotlib.pyplot as plt

import pdb
import sys
import random

sys.path.append('../')

from network import BasicNetwork, Reservoir
from utils import Bunch


colors = ['moccasin', 'springgreen', 'royalblue', 'cornflowerblue']

# plot many repetitions of a single value for g
# these are the differences between 
def plot_c():
    g = 1.5
    N = 2000
    r_noise = 0.1
    n_unique_nets = 6
    n_steps = 2000
    n_reps = 5
    all_corrs = []
    for rep in range(n_unique_nets):
        b = Bunch(N=N, Z=1, res_init_g=g, bias=False)
        net = Reservoir(b)
        corrs = []
        for j in range(n_reps):
            init_x = np.random.normal(0, 1, (1, N))
            net.reset(res_state=init_x)
            xs = np.zeros(n_steps)
            for i in range(n_steps):
                out = net()
                xs[i] = out[0,0].item()

            corr = np.correlate(xs, xs, 'full')[n_steps:]
            
            corrs.append(corr)

        all_corrs.append(corrs)
        # use this for getting the mean for each rep
        # all_corrs.append(np.mean(corrs, axis=0))

    plt.figure(figsize=(15,8))
    for rep in range(n_unique_nets):
        ax = plt.subplot(2, 3, rep+1)
        # ax.plot(all_corrs[rep], lw=2)
        for j in range(n_reps):
            ax.plot(all_corrs[rep][j], lw=2)

        ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
        ax.tick_params(axis='both', color='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('distance')
        ax.set_xlabel('timestep')

    plt.suptitle(f'g = {g}')
    plt.show()


if __name__ == '__main__':
    plot_c()
