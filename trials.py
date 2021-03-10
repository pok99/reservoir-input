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


# given info about the trial, create the target in np form
def get_y(trial):
    if trial['trial_type'].startswith('rsg'):
        y = np.arange(trial['trial_len'])
        slope = 1 / trial['t_p']
        y = y * slope - trial['rsg'][1] * slope
        y = np.clip(y, 0, 1.5)
    return y

def get_x(trial, args):
    if trial['trial_type'].startswith('rsg'):
        x = np.zeros((5, trial['trial_len']))
        rt, st, gt = trial['rsg']
        plen = trial['pulse_len']
        x[0, rt:rt+plen] = 1
        if args.separate_signal:
            x[1, st:st+plen] = 1
        else:
            x[0, st:st+plen] = 1
    return x

# toy ready set go dataset
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

            # ready / set come in thru separate channels
            # training should default to combining them
            # disable combining via the --separate_signal flag
            # trial_x = np.zeros((t_len, 2))
            # trial_x[ready_time:ready_time+p_len, 0] = 1
            # trial_x[set_time:set_time+p_len, 1] = 1

        # elif t_type == 'rsg-2c':
        #     # fixing the bug where distribution of start points is different for different intervals
        #     p_len = get_args_val(trial_args, 'plen', 5, int)
        #     max_ready = get_args_val(trial_args, 'max_ready', 80, int)
        #     args.pulse_len = p_len
        #     args.max_ready_time = max_ready
        #     if args.intervals is None or args.intervals2 is None:
        #         min_t = get_args_val(trial_args, 'gt', p_len * 4, int)
        #         max_t = get_args_val(trial_args, 'lt', t_len // 2 - max_ready // 2 - p_len * 4, int)
        #         mean_t = (min_t + max_t) / 2
        #         args.min_t = min_t
        #         args.max_t = max_t
        #     for n in range(n_trials):
        #         context = random.choice([0, 1])
        #         if context == 0:
        #             if args.intervals is None:
        #                 t_p = np.random.randint(min_t, mean_t)
        #             else:
        #                 t_p = random.choice(args.intervals)
        #         else:
        #             if args.intervals2 is None:
        #                 t_p = np.random.randint(mean_t, max_t)
        #             else:
        #                 t_p = random.choice(args.intervals2)

        #         ready_time = np.random.randint(p_len * 2, max_ready)
        #         set_time = ready_time + t_p
        #         go_time = set_time + t_p
        #         assert go_time < t_len

        #         # 2 channels for ready / set will be combined later
        #         trial_x = np.zeros((t_len, 3))
        #         trial_x[ready_time:ready_time+p_len, 0] = 1
        #         trial_x[set_time:set_time+p_len, 1] = 1
        #         trial_x[:,2] = context

        #         trial_y = np.arange(t_len)
        #         slope = 1 / t_p
        #         trial_y = trial_y * slope - set_time * slope
        #         trial_y = np.clip(trial_y, 0, 1.5)

        #         info = (ready_time, set_time, go_time, context)
        #         trials.append((trial_x, trial_y, info))

    elif t_type.startswith('copy'):
        delay = get_args_val(trial_args, 'delay', 0, int)
        config.delay = delay

        n_freqs = get_args_val(trial_args, 'n_freqs', 15, int)
        f_range = get_args_val(trial_args, 'f_range', [5, 30], float, n_vals=2)
        amp = get_args_val(trial_args, 'amp', 1, float)
        config['n_freqs'] = n_freqs
        config['f_range'] = f_range
        config['amp'] = amp


        x_r = np.arange(t_len)

        for n in range(n_trials):
            x = np.zeros((t_len))
            freqs = np.random.uniform(f_range[0], f_range[1], (n_freqs))
            amps = np.random.uniform(-amp, amp, (n_freqs))
            for i in range(n_freqs):
                x = x + amps[i] * np.sin(1/freqs[i] * x_r)

            y = np.zeros(t_len)
            if delay == 0:
                y = x
            else:
                y[delay:] = x[:-delay]

            trials.append((x, y, delay))

    elif t_type == 'delay-copy':
        n_freqs = get_args_val(trial_args, 'n_freqs', 20, int)
        f_range = get_args_val(trial_args, 'f_range', [10, 40], float, n_vals=2)
        amp = get_args_val(trial_args, 'amp', 1, float)
        config['n_freqs'] = n_freqs
        config['f_range'] = f_range
        config['amp'] = amp

        s_len = t_len // 2
        x_r = np.arange(s_len)

        for n in range(n_trials):
            x = np.zeros((t_len))
            freqs = np.random.uniform(f_range[0], f_range[1], (n_freqs))
            amps = np.random.uniform(-amp, amp, (n_freqs))
            for i in range(n_freqs):
                x[:s_len] = x[:s_len] + amps[i] * np.sin(1/freqs[i] * x_r) / np.sqrt(n_freqs)

            y = np.zeros(t_len)
            y[s_len:] = x[:s_len]

            trials.append((x, y, s_len))

    else:
        raise NotImplementedError

    return trials, args

def save_dataset(dset, name, config=None):
    fname = os.path.join('datasets', name + '.json')
    with open(fname, 'w') as f:
        json.dump(dset, f, indent=2)
    gname = os.path.join('datasets', 'configs', name + '.json')
    if config is not None:
        with open(gname, 'w') as f:
            json.dump(config.to_json(), f, indent=2)

# turn trial_args argument into usable argument variables
def get_trial_args(args):
    targ_arr = args.trial_args
    targs = Bunch()
    if args.trial_type.startswith('rsg'):
        targs.p_len = get_targs_val(targ_arr, 'pl', 5, int)
        targs.max_ready = get_targs_val(targ_arr, 'max_ready', 80, int)
        if args.intervals is None:
            targs.min_t = get_targs_val(targ_arr, 'gt', targs.p_len * 4, int)
            targs.max_t = get_targs_val(targ_arr, 'lt', args.t_len // 2 - targs.p_len * 4, int)

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
        dset = json.load(open(args.name))
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




            # if dset_type.startswith('rsg'):
            #     sample_sum = sample[i][0][:,0] + sample[i][0][:,1]
            #     ml, sl, bl = ax.stem(dset_range, sample_sum, use_line_collection=True, linefmt='coral', label='ready/set')
            #     ml.set_markerfacecolor('coral')
            #     ml.set_markeredgecolor('coral')
            #     if dset_type == 'rsg-bin':
            #         ml, sl, bl = ax.stem(dset_range, sample[i][1], use_line_collection=True, linefmt='dodgerblue', label='go')
            #         ml.set_markerfacecolor('dodgerblue')
            #         ml.set_markeredgecolor('dodgerblue')
            #     else:
            #         ax.plot(dset_range, sample[i][1], color='dodgerblue', label='go', lw=2)
            #         if dset_type == 'rsg-2c':
            #             ax.plot(dset_range, sample[i][0][:,2], color='violet', label='context', ls='--', lw=1)

            #     ax.set_ylim([-.5, 2.5])

            elif dset_type.startswith('copy'):
                if len(sample[i][0].shape) > 1:
                    ml, sl, bl = ax.stem(dset_range, sample[i][0][:,1], use_line_collection=True, linefmt='coral', label='fn')
                    ml.set_markerfacecolor('coral')
                    ml.set_markeredgecolor('coral')
                    ax.plot(dset_range, sample[i][0][:,0], color='coral', alpha=1, lw=1)
                else:
                    ax.plot(dset_range, sample[i][0], color='coral', alpha=1, lw=1)
                ax.plot(dset_range, sample[i][1], color='dodgerblue', lw=1)

        handles, labels = ax.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='lower center')
        plt.show()
