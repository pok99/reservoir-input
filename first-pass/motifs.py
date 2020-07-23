import numpy as np
import matplotlib.pyplot as plt

import pickle
import json
import random
import pdb
import os
import argparse
import sys

from sklearn.gaussian_process import GaussianProcessRegressor as gpr

from utils import lrange, load_rb
from helpers import get_dim

# from classes import Trajectory

eps = 1e-6

# default parameters for generating motifs, for different # of anchor pts
# [min t distance bw pts, uniform t interval to pick next pt, gaussian x interval variance]
TX_PARAMS = {
    0: [3],
    1: [0.2, 2, 4],
    2: [0.5, 1.5, 3],
    3: [0.5, 1.5, 3]
}


def gen_motifs(n, l_range=[5, 30], n_freqs=15, f_range=[3, 30], amp=1, start_zero=True):
    config = {}
    config['start_zero'] = start_zero
    config['l_range'] = l_range
    config['n_freqs'] = n_freqs
    config['f_range'] = f_range
    config['amp'] = amp

    motifs = []

    for i in range(n):

        len_motif = np.random.randint(l_range[0], l_range[1])
        x = np.arange(len_motif)
        y = np.zeros(len_motif)

        freqs = np.random.uniform(f_range[0], f_range[1], (n_freqs))
        amps = np.random.uniform(-amp, amp, (n_freqs))
        for i in range(n_freqs):
            y = y + amps[i] * np.cos(1/freqs[i] * x)
        
        if start_zero:
            z = 1 / len_motif * x
            y *= z

        motifs.append(y)

    return (motifs, config)

def plot_motifs(motifs, name=None, labels=True):
    dim = get_dim(motifs[0][0])
    if dim == 1:
        plt.style.use('ggplot')
        for ind, m in enumerate(motifs):
            t = np.arange(len(m))
            plt.plot(t, m, lw=2, label=ind)
        if labels:
            if len(motifs) < 15:
                plt.legend()

        plt.xlabel('timestep')
        plt.ylabel('activity')
        plt.show()

    # elif dim == 3:
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #     for m in motifs:
    #         t = lrange(len(m))
    #         xp = m.x[:,0]
    #         yp = m.x[:,1]
    #         zp = m.x[:,2]

    #         ax.plot3D(xp, yp, zp, lw=1)

    #     ax.grid(False)
    #     plt.axis('off')
    #     z100 = np.zeros(100)
    #     a100 = np.linspace(-5, 5, 100)
    #     ax.plot3D(z100, z100, a100, color='black')
    #     ax.plot3D(z100, a100, z100, color='black')
    #     ax.plot3D(a100, z100, z100, color='black')

    #     ax.set_xlim3d([-5, 5])
    #     ax.set_ylim3d([-5, 5])
    #     ax.set_zlim3d([-5, 5])

    #     plt.show()

def save_motifs(motifs, name, config):
    with open(os.path.join('motifsets', name+'.pkl'), 'wb') as f:
        pickle.dump(motifs, f)

    with open(os.path.join('motifsets', 'configs', name+'.json'), 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('--name', type=str, default='temp')
    parser.add_argument('-n', type=int, default=10)
    parser.add_argument('--l_range', nargs=2, type=int, default=[5, 20])
    parser.add_argument('--f_range', nargs=2, type=int, default=[2, 30])
    parser.add_argument('--amp', type=float, default=1)
    parser.add_argument('--n_freqs', type=int, default=15)
    parser.add_argument('--dim', type=int, default=1)
    parser.add_argument('--start_nonzero', action='store_true')
    args = parser.parse_args()

    args.start_zero = not args.start_nonzero


    if args.mode == 'create':
        motifs, config = gen_motifs(
            args.n,
            l_range=args.l_range,
            n_freqs=args.n_freqs,
            f_range=args.f_range,
            amp=args.amp,
            start_zero=args.start_zero)
        save_motifs(motifs, args.name, config)

    elif args.mode == 'load':
        motifs = load_rb(args.name)

    elif args.mode == 'test':
        motifs, config = gen_motifs(
            args.n,
            l_range=args.l_range,
            n_freqs=args.n_freqs,
            f_range=args.f_range,
            amp=args.amp,
            start_zero=args.start_zero)
        save_motifs(motifs, args.name, config)

    plot_motifs(motifs, name=args.name, labels=False)
