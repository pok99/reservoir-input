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

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import collections as matcoll

import argparse

# from motifs import gen_fn_motifs
from utils import add_config_args, load_rb, Bunch

eps = 1e-6

mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.linewidth'] = .5

# toy ready set go dataset
def create_dataset(args):
    name = args.name
    t_type = args.trial_type
    n_trials = args.n_trials
    t_len = args.trial_len
    trial_args = args.trial_args

    trials = []
    if t_type.startswith('rsg'):
        # should generally not use rsg1 and use rsg instead
        # old implementation with bug here for reproducibility
        if t_type.startswith('rsg1'):
            p_len = get_args_val(trial_args, 'plen', 5, int)
            args.pulse_len = p_len
            if args.intervals is None:
                # amount of time in between ready and set cues
                min_t = get_args_val(trial_args, 'gt', p_len * 4, int)
                max_t = get_args_val(trial_args, 'lt', t_len // 2 - p_len * 4, int)
                args.min_t = min_t
                args.max_t = max_t
            for n in range(n_trials):
                if args.intervals is None:
                    t_p = np.random.randint(min_t, max_t)
                else:
                    # use one of the intervals that we desire
                    num = random.choice(args.intervals)
                    assert num < t_len / 2
                    t_p = num

                ready_time = np.random.randint(p_len * 4, t_len - t_p * 2 - p_len * 4)
                set_time = ready_time + t_p
                go_time = set_time + t_p

                assert go_time < t_len

                trial_x = np.zeros((t_len, 2))
                trial_x[ready_time:ready_time+p_len, 0] = 1
                trial_x[set_time:set_time+p_len, 1] = 1

                trial_y = np.zeros((t_len))
                if t_type == 'rsg1-bin':
                    trial_y[go_time:go_time+p_len] = 1
                elif t_type == 'rsg1-sohn':
                    # A = 3
                    # alpha = (t_p - p_len) / t_p / np.log(4/3)
                    # TODO: FIX THIS BUG WITH OFFSETS
                    trial_y_temp = np.arange(t_len - set_time - p_len)
                    trial_y_fn = lambda t: t / t_p
                    trial_y[set_time+p_len:] = trial_y_fn(trial_y_temp)
                    trial_y = np.clip(trial_y, 0, 2)
                elif t_type == 'rsg1-window':
                    trial_x = np.zeros((ready_time + 3 * t_p, 2))
                    trial_x[ready_time:ready_time+p_len, 0] = 1
                    trial_x[set_time:set_time+p_len, 1] = 1
                    trial_y = np.zeros((ready_time + 3 * t_p))

                    A = 3
                    alpha = (t_p - p_len) / t_p / np.log(4/3)
                    trial_y_temp = np.arange(2 * t_p - p_len)
                    trial_y_fn = lambda t: A * (np.exp(t / (alpha * t_p)) - 1)
                    trial_y[set_time+p_len:] = trial_y_fn(trial_y_temp)

                info = (ready_time, set_time, go_time)
                trials.append((trial_x, trial_y, info))

        elif t_type.startswith('rsg2'):
            # fixed the bug where distribution of start points is different for different intervals
            p_len = get_args_val(trial_args, 'plen', 5, int)
            max_ready = get_args_val(trial_args, 'max_ready', 80, int)
            args.pulse_len = p_len
            args.max_ready_time = max_ready
            if args.intervals is None:
                min_t = get_args_val(trial_args, 'gt', p_len * 4, int)
                max_t = get_args_val(trial_args, 'lt', t_len // 2 - p_len * 4, int)
                args.min_t = min_t
                args.max_t = max_t
            for n in range(n_trials):
                if args.intervals is None:
                    t_p = np.random.randint(min_t, max_t)
                else:
                    t_p = random.choice(args.intervals)
                ready_time = np.random.randint(p_len * 2, max_ready)
                set_time = ready_time + t_p
                go_time = set_time + t_p

                assert go_time < t_len

                trial_x = np.zeros((t_len, 2))
                trial_x[ready_time:ready_time+p_len, 0] = 1
                trial_x[set_time:set_time+p_len, 1] = 1

                trial_y = np.arange(t_len)
                slope = 1 / t_p
                trial_y = trial_y * slope - set_time * slope
                trial_y = np.clip(trial_y, 0, 2)

                info = (ready_time, set_time, go_time)
                trials.append((trial_x, trial_y, info))

        elif t_type == 'rsg-2c':
            # fixing the bug where distribution of start points is different for different intervals
            p_len = get_args_val(trial_args, 'plen', 5, int)
            max_ready = get_args_val(trial_args, 'max_ready', 80, int)
            args.pulse_len = p_len
            args.max_ready_time = max_ready
            if args.intervals is None or args.intervals2 is None:
                min_t = get_args_val(trial_args, 'gt', p_len * 4, int)
                max_t = get_args_val(trial_args, 'lt', t_len // 2 - max_ready // 2 - p_len * 4, int)
                mean_t = (min_t + max_t) / 2
                args.min_t = min_t
                args.max_t = max_t
            for n in range(n_trials):
                context = random.choice([0, 1])
                if context == 0:
                    if args.intervals is None:
                        t_p = np.random.randint(min_t, mean_t)
                    else:
                        t_p = random.choice(args.intervals)
                else:
                    if args.intervals2 is None:
                        t_p = np.random.randint(mean_t, max_t)
                    else:
                        t_p = random.choice(args.intervals2)

                ready_time = np.random.randint(p_len * 2, max_ready)
                set_time = ready_time + t_p
                go_time = set_time + t_p
                assert go_time < t_len

                trial_x = np.zeros((t_len, 2))
                trial_x[ready_time:ready_time+p_len, 0] = (context + 1) / 2
                trial_x[set_time:set_time+p_len, 1] = 1

                trial_y = np.arange(t_len)
                slope = 1 / t_p
                trial_y = trial_y * slope - set_time * slope
                trial_y = np.clip(trial_y, 0, 2)

                info = (ready_time, set_time, go_time, context)
                trials.append((trial_x, trial_y, info))

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

def get_args_val(args, name, default, dtype, n_vals=1):
    if name in args:
        idx = args.index(name)
        if n_vals == 1:
            val = dtype(args[idx + 1])
        else:
            vals = []
            for i in range(1, n_vals+1):
                vals.append(dtype(args[idx + i]))
    else:
        val = default
    return val

def save_dataset(dset, name, config=None):
    fname = name + '.pkl'
    with open(os.path.join('datasets', fname), 'wb') as f:
        pickle.dump(dset, f)
    gname = name + '.json'
    if config is not None:
        with open(os.path.join('datasets', 'configs', gname), 'w') as f:
            json.dump(config.to_json(), f, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='load', choices=['create', 'load'])
    parser.add_argument('name')
    parser.add_argument('-c', '--config', default=None, help='create from a config file')
    parser.add_argument('-t', '--trial_type', default='rsg-2c', help='type of trial to create')
    parser.add_argument('-i', '--intervals', nargs='*', type=int, default=None, help='rsg intervals for context 1')
    parser.add_argument('-j', '--intervals2', nargs='*', type=int, default=None, help='rsg intervals for context 2 (rsg-2c)')
    parser.add_argument('--motifs', type=str, help='path to motifs')
    parser.add_argument('-a', '--trial_args', nargs='*', help='terms to specify parameters of trial type')
    parser.add_argument('-l', '--trial_len', type=int, default=500)
    parser.add_argument('-n', '--n_trials', type=int, default=2000)
    args = parser.parse_args()
    args = add_config_args(args, args.config)

    args.argv = ' '.join(sys.argv)

    if args.trial_args is None:
        args.trial_args = []

    if args.mode == 'create':
        dset, config = create_dataset(args)
        save_dataset(dset, args.name, config=config)
    elif args.mode == 'load':
        dset = load_rb(args.name)

        # all for getting config path to get dataset type
        dset_path_tmp = args.name.split('/')
        config_path_tmp = '/'.join(dset_path_tmp[:-1] + ['configs'] + [dset_path_tmp[-1]])
        config_path = config_path_tmp[:-4] + '.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        dset_type = config['trial_type']

        dset_len = len(dset)
        sample = random.sample(dset, 12)
        dset_range = range(len(sample[0][0]))
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

            dset_range = range(len(sample[i][0]))
            if dset_type.startswith('rsg'):
                sample_sum = sample[i][0][:,0] + sample[i][0][:,1]
                ml, sl, bl = ax.stem(dset_range, sample_sum, use_line_collection=True, linefmt='coral', label='ready/set')
                ml.set_markerfacecolor('coral')
                ml.set_markeredgecolor('coral')
                if dset_type == 'rsg-bin':
                    ml, sl, bl = ax.stem(dset_range, sample[i][1], use_line_collection=True, linefmt='dodgerblue', label='go')
                    ml.set_markerfacecolor('dodgerblue')
                    ml.set_markeredgecolor('dodgerblue')
                else:
                    ax.plot(dset_range, sample[i][1], color='dodgerblue', label='go', lw=2)

                ax.set_ylim([-.5, 2.5])

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
