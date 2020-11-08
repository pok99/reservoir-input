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
from matplotlib import collections as matcoll

import argparse

from motifs import gen_fn_motifs
from utils import load_rb

eps = 1e-6

# toy ready set go dataset
def create_dataset(args):
    name = args.name
    t_type = args.trial_type
    n_trials = args.n_trials
    t_len = args.trial_len
    trial_args = args.trial_args

    config = {}
    config['argv'] = sys.argv
    config['t_type'] = t_type
    config['n_trials'] = n_trials
    config['t_len'] = t_len

    trials = []

    if t_type == 'rsg-gaussian':
        '''
        trial_args options:
        - use_ints
        - lt [x]
        - gt [x]
        - scale [x]

        '''
        # use_ints is overridden by rsg_intervals
        use_ints = 'use_ints' in trial_args
        if args.rsg_intervals is None:
            # amount of time in between ready and set cues
            min_t = get_args_val(trial_args, 'gt', 15, int)
            max_t = get_args_val(trial_args, 'lt', t_len // 2 - 15, int)
            config['min_t'] = min_t
            config['max_t'] = max_t
        for n in range(n_trials):
            if args.rsg_intervals is None:
                if use_ints:
                    t_p = np.random.randint(min_t, max_t)
                else:
                    t_p = np.round(np.random.uniform(min_t, max_t), 2)
            else:
                # use one of the intervals that we desire
                # overrides use_ints
                num = random.choice(args.rsg_intervals)
                assert num < t_len / 2
                t_p = num

            if use_ints:
                ready_time = np.random.randint(5, t_len - t_p * 2 - 10)
            else:
                ready_time = np.round(np.random.uniform(5, t_len - t_p * 2 - 10), 2)
                
            set_time = ready_time + t_p
            go_time = set_time + t_p

            # check if width of gaussian is changed from default
            scale = get_args_val(trial_args, 'scale', 1, float)
            config['scale'] = scale

            trial_range = np.arange(t_len)
            trial_x = norm.pdf(trial_range, loc=ready_time, scale=1)
            trial_x += norm.pdf(trial_range, loc=set_time, scale=1)
            # scaling by `scale` so the height of the middle is always the same
            trial_y = 2 * scale * norm.pdf(trial_range, loc=go_time, scale=scale)

            info = (ready_time, set_time, go_time)

            trials.append((trial_x, trial_y, info))

    elif t_type == 'rsg-pulse':
        pulse_len = get_args_val(trial_args, 'plen', 5, int)
        config['pulse_len'] = pulse_len
        if args.rsg_intervals is None:
            # amount of time in between ready and set cues
            min_t = get_args_val(trial_args, 'gt', 15, int)
            max_t = get_args_val(trial_args, 'lt', t_len // 2 - 15, int)
            config['min_t'] = min_t
            config['max_t'] = max_t
        for n in range(n_trials):
            if args.rsg_intervals is None:
                t_p = np.random.randint(min_t, max_t)
            else:
                # use one of the intervals that we desire
                num = random.choice(args.rsg_intervals)
                assert num < t_len / 2
                t_p = num

            ready_time = np.random.randint(5, t_len - t_p * 2 - 10)
                
            set_time = ready_time + t_p
            go_time = set_time + t_p

            trial_x = np.zeros((t_len))
            trial_y = np.zeros((t_len))
            trial_x[ready_time:ready_time+pulse_len] = 1
            trial_x[set_time:set_time+pulse_len] = 1
            trial_y[go_time:go_time+pulse_len] = 1

            info = (ready_time, set_time, go_time)

            trials.append((trial_x, trial_y, info))

    elif t_type == 'rsg-pulse2d':
        pulse_len = get_args_val(trial_args, 'plen', 5, int)
        config['pulse_len'] = pulse_len
        if args.rsg_intervals is None:
            # amount of time in between ready and set cues
            min_t = get_args_val(trial_args, 'gt', 15, int)
            max_t = get_args_val(trial_args, 'lt', t_len // 2 - 15, int)
            config['min_t'] = min_t
            config['max_t'] = max_t
        for n in range(n_trials):
            if args.rsg_intervals is None:
                t_p = np.random.randint(min_t, max_t)
            else:
                # use one of the intervals that we desire
                num = random.choice(args.rsg_intervals)
                assert num < t_len / 2
                t_p = num

            ready_time = np.random.randint(5, t_len - t_p * 2 - 10)
                
            set_time = ready_time + t_p
            go_time = set_time + t_p

            trial_x = np.zeros((t_len, 2))
            trial_y = np.zeros((t_len))
            trial_x[ready_time:ready_time+pulse_len, 0] = 1
            trial_x[set_time:set_time+pulse_len, 1] = 1
            trial_y[go_time:go_time+pulse_len] = 1

            info = (ready_time, set_time, go_time)

            trials.append((trial_x, trial_y, info))

    elif t_type == 'rsg-sohn':
        pulse_len = get_args_val(trial_args, 'plen', 5, int)
        config['pulse_len'] = pulse_len
        if args.rsg_intervals is None:
            # amount of time in between ready and set cues
            min_t = get_args_val(trial_args, 'gt', 15, int)
            max_t = get_args_val(trial_args, 'lt', t_len // 2 - 15, int)
            config['min_t'] = min_t
            config['max_t'] = max_t
        for n in range(n_trials):
            if args.rsg_intervals is None:
                t_p = np.random.randint(min_t, max_t)
            else:
                # use one of the intervals that we desire
                num = random.choice(args.rsg_intervals)
                assert num < t_len / 2
                t_p = num

            ready_time = np.random.randint(5, t_len - t_p * 2 - 10)
                
            set_time = ready_time + t_p
            go_time = set_time + t_p

            trial_x = np.zeros(t_len)
            trial_y = np.zeros(t_len)
            trial_x[ready_time:ready_time+pulse_len] = 1
            trial_x[set_time:set_time+pulse_len] = 1
            trial_y[go_time:go_time+pulse_len] = 1
            trial_y_temp = np.arange(t_len - set_time)
            trial_y_fn = lambda t: 1 / (t_p - pulse_len) * t
            trial_y[set_time:] = trial_y_fn(trial_y_temp)
            trial_y = np.clip(trial_y, 0, 2)

            info = (ready_time, set_time, go_time)

            trials.append((trial_x, trial_y, info))


    elif t_type.startswith('copy'):
        delay = get_args_val(trial_args, 'delay', 0, int)
        config['delay'] = delay
        
        dim = 1
        x = np.arange(0, t_len)
        ys = []

        if t_type == 'copy':
            n_freqs = get_args_val(trial_args, 'n_freqs', 15, int)
            f_range = get_args_val(trial_args, 'f_range', [2, 30], float, n_vals=2)
            amp = get_args_val(trial_args, 'amp', 1, float)
            start_zero = 'start_nonzero' not in trial_args
            config['n_freqs'] = n_freqs
            config['f_range'] = f_range
            config['amp'] = amp
            config['start_zero'] = start_zero

            fn = np.sin if start_zero else np.cos

            for n in range(n_trials):
                y = np.zeros_like(x)
                freqs = np.random.uniform(f_range[0], f_range[1], (n_freqs))
                amps = np.random.uniform(-amp, amp, (n_freqs))
                for i in range(n_freqs):
                    y = y + amps[i] * fn(1/freqs[i] * x)

                ys.append(y)

        elif t_type == 'copy_motifs':
            assert args.motifs is not None
            motifs = load_rb(args.motifs)
            pause = get_args_val(trial_args, 'pause', 10, int)
            amp = get_args_val(trial_args, 'amp', .1, float)
            config['pause'] = pause
            config['amp'] = amp

            for n in range(n_trials):
                y = gen_fn_motifs(motifs, length=t_len, pause=pause, amp=amp, smoothing='cubic')
                ys.append(y)
        else:
            raise Exception

        for y in ys:
            z = np.zeros_like(y)
            if delay == 0:
                z = y
            else:
                z[delay:] = y[:-delay]

            trials.append((y, z, delay))

    elif t_type == 'amplify':
        n_freqs = get_args_val(trial_args, 'n_freqs', 15, int)
        f_range = get_args_val(trial_args, 'f_range', [3, 30], float, n_vals=2)
        amp = get_args_val(trial_args, 'amp', 1, float)
        mag = get_args_val(trial_args, 'mag', 1, float)
        config['n_freqs'] = n_freqs
        config['f_range'] = f_range
        config['amp'] = amp
        config['mag'] = mag

        for n in range(n_trials):
            x = np.arange(0, t_len)
            y = np.zeros_like(x)

            freqs = np.random.uniform(f_range[0], f_range[1], (n_freqs))
            amps = np.random.uniform(-amp, amp, (n_freqs))
            for i in range(n_freqs):
                y = y + amps[i] * np.cos(1/freqs[i] * x)

            z = y * mag
            trials.append((y, z, mag))

    return trials, config

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
            json.dump(config, f, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='load')
    parser.add_argument('name')
    parser.add_argument('-t', '--trial_type', default='rsg-pulse')
    parser.add_argument('-i', '--rsg_intervals', nargs='*', type=int, default=None)
    parser.add_argument('--motifs', type=str, help='path to motifs')
    parser.add_argument('-a', '--trial_args', nargs='*', help='terms to specify parameters of trial type')
    parser.add_argument('-l', '--trial_len', type=int, default=200)
    parser.add_argument('-n', '--n_trials', type=int, default=1000)
    args = parser.parse_args()

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
        dset_type = config['t_type']

        dset_len = len(dset)
        sample = random.sample(dset, 6)
        dset_range = range(len(sample[0][0]))
        fig, ax = plt.subplots(2,3,sharex=True, sharey=True, figsize=(8,4))
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

            if dset_type == 'rsg' or dset_type == 'rsg-gaussian' or dset_type == 'rsg-sohn':
                ax.plot(dset_range, sample[i][0], color='coral', label='ready/set', lw=2)
                ax.plot(dset_range, sample[i][1], color='dodgerblue', label='go', lw=2)
            elif dset_type == 'rsg-pulse':
                ml, sl, bl = ax.stem(dset_range, sample[i][0], use_line_collection=True, linefmt='coral', label='ready/set')
                ml.set_markerfacecolor('coral')
                ml.set_markeredgecolor('coral')
                ml.set_markersize(3)
                sl.set_linewidth(.5)
                ml, sl, bl = ax.stem(dset_range, sample[i][1], use_line_collection=True, linefmt='dodgerblue', label='ready/set')
                ml.set_markerfacecolor('dodgerblue')
                ml.set_markeredgecolor('dodgerblue')
                ml.set_markersize(3)
                sl.set_linewidth(.5)
            elif dset_type == 'rsg-pulse2d':
                ml, sl, bl = ax.stem(dset_range, sample[i][0][:,0], use_line_collection=True, linefmt='coral', label='ready/set')
                ml.set_markerfacecolor('coral')
                ml.set_markeredgecolor('coral')
                ml.set_markersize(3)
                sl.set_linewidth(.5)
                ml, sl, bl = ax.stem(dset_range, sample[i][0][:,1], use_line_collection=True, linefmt='coral', label='ready/set')
                ml.set_markerfacecolor('coral')
                ml.set_markeredgecolor('coral')
                ml.set_markersize(3)
                sl.set_linewidth(.5)
                ml, sl, bl = ax.stem(dset_range, sample[i][1], use_line_collection=True, linefmt='dodgerblue', label='ready/set')
                ml.set_markerfacecolor('dodgerblue')
                ml.set_markeredgecolor('dodgerblue')
                ml.set_markersize(3)
                sl.set_linewidth(.5)

            ax.set_ylim([-.5, 2.5])

        handles, labels = ax.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='lower center')
        plt.show()
    # confirm ready set go works
    # for i in range(5):
    #     np.random.shuffle(dset)
    #     print(np.where(dset[i][0] == 1), np.argmax(dset[i][1]))
