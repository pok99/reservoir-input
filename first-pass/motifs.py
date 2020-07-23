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

eps = 1e-6


def gen_fn(length=50, amp=1, n_freqs=15, f_range=[3,30], start_zero=True):
    x = np.arange(length)
    y = np.zeros(length)

    fn = np.sin if start_zero else np.cos

    freqs = np.random.uniform(f_range[0], f_range[1], (n_freqs))
    amps = np.random.uniform(-amp, amp, (n_freqs))
    for i in range(n_freqs):
        y = y + amps[i] * fn(1/freqs[i] * x)

    return y

def gen_motifs(n, l_range=[5, 30], amp=1, n_freqs=15, f_range=[3, 30], start_zero=True):
    config = {}
    config['start_zero'] = start_zero
    config['l_range'] = l_range
    config['n_freqs'] = n_freqs
    config['f_range'] = f_range
    config['amp'] = amp

    motifs = []

    for i in range(n):
        len_motif = np.random.randint(l_range[0], l_range[1])
        y = gen_fn(len_motif, amp=amp, n_freqs=n_freqs, f_range=f_range, start_zero=start_zero)
        motifs.append(y)

    return (motifs, config)

# create fn from motifs
def gen_fn_motifs(motifs, length=200, pause=10, amp=.1, smoothing='cubic'):

    cur_y = 0
    prev_slope = 0
    y = np.array([])
    while len(y) < length:
        cm = np.random.choice(motifs)
        cm = cm[1:]
        if smoothing is not None:
            if len(y) > 0:
                if smoothing == 'cubic':
                    cur_slope = cm[1]
                    x1 = len(y) - 1
                    x2 = len(y) + pause - 1
                    # solve set of linear equations with conditions:
                    # slopes on both sides are as given
                    # starts at end of first motifs, ends at 0
                    M = np.array([
                        [x1 ** 3, x1 ** 2, x1, 1],
                        [x2 ** 3, x2 ** 2, x2, 1],
                        [3 * x1 ** 2, 2 * x1, 1, 0],
                        [3 * x2 ** 2, 2 * x2, 1, 0]])
                    b = np.array([y[-1], 0, prev_slope, cur_slope])
                    coefs = np.linalg.solve(M, b)
                    x = np.arange(len(y), len(y) + pause)
                    y_x = coefs[0] * x ** 3 + coefs[1] * x ** 2 + coefs[2] * x + coefs[3]

                elif smoothing == 'quadratic':
                    # similar to above, but you don't get quadratic ending at 0

                    cur_slope = cm[1]
                    # calculate coefficients of quadratic
                    a = (cur_slope - prev_slope) / (2 * pause)
                    b = prev_slope - 2 * a * x1
                    c = y[-1] - a * x1 ** 2 - b * x1
                    # create quadratic curve
                    x = np.arange(len(y), len(y) + pause)
                    y_x = a * x ** 2 + b * x + c

                y = np.concatenate((y, y_x))
                cur_y = y[-1]
            prev_slope = cm[-1] - cm[-2]
        y = np.concatenate((y, cm + cur_y))
        cur_y = y[-1]

    y = y[:length]

    if amp != 0:
        z = y + gen_fn(length, amp=amp, n_freqs=10)

    return z



def plot_fn():

    plt.style.use('ggplot')
    motifs = load_rb('motifsets/test.pkl')
    traj = gen_fn_motifs(motifs, length=200, amp=.1, smoothing='cubic')
    #traj = gen_fn(length=200, amp=.1, n_freqs=10)
    plt.plot(traj)
    
    plt.xlabel('timestep')
    plt.ylabel('activity')
    plt.show()


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
    #parser.add_argument('--motifs', type=str)
    parser.add_argument('--start_nonzero', action='store_true')
    args = parser.parse_args()

    args.start_zero = not args.start_nonzero

    if args.mode == 'fn':
        plot_fn()

    else:
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
