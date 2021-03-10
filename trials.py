import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import pickle
import os
import sys
import json
import pdb
import random
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import collections as matcoll

import argparse

# from motifs import gen_fn_motifs
from utils import update_args, get_file_args, load_rb, Bunch

eps = 1e-6

mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.linewidth'] = .5


# given info about the trial, create the x and y in np form
def get_x(trial, args=None):
    if trial['trial_type'].startswith('rsg'):
        x = np.zeros((5, trial['trial_len']))
        rt, st, gt = trial['rsg']
        plen = trial['pulse_len']
        x[0, rt:rt+plen] = 1
        if args is not None and args.separate_signal:
            x[1, st:st+plen] = 1
        else:
            x[0, st:st+plen] = 1
    elif trial['trial_type'] == 'delay-copy':
        x = trial['x']
    return x

def get_y(trial):
    if trial['trial_type'].startswith('rsg'):
        y = np.arange(trial['trial_len'])
        slope = 1 / trial['t_p']
        y = y * slope - trial['rsg'][1] * slope
        y = np.clip(y, 0, 1.5)
    elif trial['trial_type'] == 'delay-copy':
        y = trial['y']
    return y


def create_dataset(args):
    t_type = args.trial_type
    n_trials = args.n_trials
    t_len = args.t_len

    trials = []
    if t_type.startswith('rsg'):
        for n in range(n_trials):
            if args.intervals is None:
                t_p = np.random.randint(args.min_t, args.max_t)
            else:
                t_p = random.choice(args.intervals)
            ready_time = np.random.randint(args.p_len * 2, args.max_ready)
            set_time = ready_time + t_p
            go_time = set_time + t_p

            assert go_time < t_len

            trial = {
                'trial_type': 'rsg',
                'pulse_len': args.p_len,
                'trial_len': t_len,
                'rsg': (ready_time, set_time, go_time),
                't_p': t_p
            }

            trials.append(trial)

    elif t_type == 'delay-copy':
        s_len = t_len // 2
        x_r = np.arange(s_len)

        for n in range(n_trials):
            
            freqs = np.random.uniform(args.f_range[0], args.f_range[1], (args.n_freqs))
            amps = np.random.uniform(-args.amp, args.amp, (args.n_freqs))

            x = np.zeros((t_len))
            for i in range(args.n_freqs):
                x[:s_len] = x[:s_len] + amps[i] * np.sin(1/freqs[i] * x_r) / np.sqrt(args.n_freqs)

            y = np.zeros(t_len)
            y[s_len:] = x[:s_len]

            trial = {
                'trial_type': 'delay-copy',
                'trial_len': t_len,
                'x': x,
                'y': y
            }

            trials.append(trial)

    else:
        raise NotImplementedError

    return trials, args

# turn trial_args argument into usable argument variables
def get_trial_args(args):
    tarr = args.trial_args
    targs = Bunch()
    if args.trial_type.startswith('rsg'):
        targs.p_len = get_targs_val(tarr, 'pl', 5, int)
        targs.max_ready = get_targs_val(tarr, 'max_ready', 80, int)
        if args.intervals is None:
            targs.min_t = get_targs_val(tarr, 'gt', targs.p_len * 4, int)
            targs.max_t = get_targs_val(tarr, 'lt', args.t_len // 2 - targs.p_len * 4, int)

    elif args.trial_type == 'delay-copy':
        targs.n_freqs = get_targs_val(tarr, 'n_freqs', 20, int)
        targs.f_range = get_targs_val(tarr, 'f_range', [10, 40], float, n_vals=2)
        targs.amp = get_targs_val(tarr, 'amp', 1, float)

    return targs

# get particular value(s) given name and casting type
def get_targs_val(targs, name, default, dtype, n_vals=1):
    if name in targs:
        idx = targs.index(name)
        if n_vals == 1:
            val = dtype(targs[idx + 1])
        else:
            vals = []
            for i in range(1, n_vals+1):
                vals.append(dtype(targs[idx + i]))
    else:
        val = default
    return val


def save_dataset(dset, name, config=None):
    fname = os.path.join('datasets', name + '.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(dset, f)
    gname = os.path.join('datasets', 'configs', name + '.json')
    if config is not None:
        with open(gname, 'w') as f:
            json.dump(config.to_json(), f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='load', choices=['create', 'load'])
    parser.add_argument('name')
    parser.add_argument('-c', '--config', default=None, help='create from a config file')

    # general dataset arguments
    parser.add_argument('-t', '--trial_type', default='rsg', help='type of trial to create')
    parser.add_argument('-l', '--t_len', type=int, default=500)
    parser.add_argument('-n', '--n_trials', type=int, default=2000)

    # rsg specific arguments
    parser.add_argument('-i', '--intervals', nargs='*', type=int, default=None, help='select from rsg intervals')

    parser.add_argument('--motifs', type=str, help='path to motifs')
    parser.add_argument('-a', '--trial_args', nargs='*', default=[], help='terms to specify parameters of trial type')

    
    args = parser.parse_args()
    config_args = get_file_args(args.config)
    trial_args = get_trial_args(args)
    args = update_args(args, config_args)
    args = update_args(args, trial_args)

    args.argv = ' '.join(sys.argv)

    if args.mode == 'create':
        dset, config = create_dataset(args)
        save_dataset(dset, args.name, config=config)
    elif args.mode == 'load':
        dset = load_rb(args.name)
        dset_type = dset[0]['trial_type']
        dset_range = np.arange(dset[0]['trial_len'])

        samples = random.sample(dset, 12)
        fig, ax = plt.subplots(3,4,sharex=True, sharey=True, figsize=(10,6))
        for i, ax in enumerate(fig.axes):
            ax.axvline(x=0, color='dimgray', alpha = 1)
            ax.axhline(y=0, color='dimgray', alpha = 1)
            ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
            ax.tick_params(axis='both', color='white')
            #ax.set_title(sample[i][2])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            trial = samples[i]
            trial_x = get_x(trial)
            trial_y = get_y(trial)

            if dset_type.startswith('rsg'):
                trial_x = np.sum(trial_x, axis=0)
                ml, sl, bl = ax.stem(dset_range, trial_x, use_line_collection=True, linefmt='coral', label='ready/set')
                ml.set_markerfacecolor('coral')
                ml.set_markeredgecolor('coral')
                if dset_type == 'rsg-bin':
                    ml, sl, bl = ax.stem(dset_range, [1], use_line_collection=True, linefmt='dodgerblue', label='go')
                    ml.set_markerfacecolor('dodgerblue')
                    ml.set_markeredgecolor('dodgerblue')
                else:
                    ax.plot(dset_range, trial_y, color='dodgerblue', label='go', lw=2)

            elif dset_type == 'delay-copy':
                ax.plot(dset_range, trial_x, color='coral', lw=1)
                ax.plot(dset_range, trial_y, color='dodgerblue', lw=1)

        handles, labels = ax.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='lower center')
        plt.show()
