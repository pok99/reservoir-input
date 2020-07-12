import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF
import pickle
import os
import sys
import json
import pdb
import random

import matplotlib.pyplot as plt

import argparse

eps = 1e-6

# toy ready set go dataset
def create_dataset(args):
    name = args.name
    t_type = args.trial_type
    n_trials = args.n_trials
    t_len = args.trial_len
    trial_args = args.trial_args

    trials = []

    if t_type == 'rsg':
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
            min_t = 15
            max_t = t_len // 2 - 15
            if 'gt' in trial_args:
                idx = trial_args.index('gt')
                min_t = int(trial_args[idx + 1])
            elif 'lt' in trial_args:
                idx = trial_args.index('lt')
                max_t = int(trial_args[idx + 1])
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

            # output 0s and 1s instead of pdf, use with CrossEntropyLoss
            if 'delta' in trial_args:
                trial_x = np.zeros((t_len))
                trial_y = np.zeros((t_len))
                
                trial_x[ready_time-1:ready_time+2] = 1
                trial_x[set_time-1:set_time+2] = 1
                trial_y[go_time-2:go_time+3] = 1
            else:
                # check if width of gaussian is changed from default
                scale = get_args_val(trial_args, 'scale', 1)

                trial_range = np.arange(t_len)
                trial_x = norm.pdf(trial_range, loc=ready_time, scale=1)
                trial_x += norm.pdf(trial_range, loc=set_time, scale=1)
                # scaling by `scale` so the height of the middle is always the same
                trial_y = 4 * scale * norm.pdf(trial_range, loc=go_time, scale=scale)

            info = (ready_time, set_time, go_time)

            trials.append((trial_x, trial_y, info))


    elif t_type == 'copy':
        for n in range(n_trials):
            dim = 1
            x = np.arange(0, t_len)
            x_list = x[..., np.newaxis]

            interval = int(get_args_val(trial_args, 'interval', 10))
            scale = get_args_val(trial_args, 'scale', 1)
            delay = int(get_args_val(trial_args, 'delay', 0))
            rbf = RBF(length_scale=3)

            x_filter = x_list[::interval]
            n_pts = x_filter.squeeze().shape[0]

            y_filter = np.zeros((n_pts, dim))
            y_filter[0] = 0
            for i in range(1, n_pts):
                if 'smoothing' in trial_args:
                    y_filter[i] = np.random.multivariate_normal(y_filter[i-1]/2, cov=scale*np.eye(dim))
                else:
                    y_filter[i] = np.random.multivariate_normal(np.zeros(dim), cov=scale*np.eye(dim))

            gp = gpr(kernel=rbf, normalize_y=True).fit(x_filter, y_filter)
            y_prediction, y_std = gp.predict(x_list, return_std=True)

            y = y_prediction.reshape(-1)

            z = np.zeros_like(y)
            if delay == 0:
                z = y
            else:
                z[delay:] = y[:-delay]

            trials.append((y, z, delay))

    return trials

def get_args_val(args, name, default):
    if name in args:
        idx = args.index(name)
        val = float(args[idx + 1])
    else:
        val = default
    return val

def save_dataset(dset, name, args=None):
    fname = name + '.pkl'
    with open(os.path.join('datasets', fname), 'wb') as f:
        pickle.dump(dset, f)
    # gname = name + '.json'
    # if args is not None:
    #     args = vars(args)
    #     with open(os.path.join('data', gname), 'w') as f:
    #         json.dump(args, f)

def load_dataset(fpath):
    with open(fpath, 'rb') as f:
        dset = pickle.load(f)
    return dset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='load')
    parser.add_argument('name')
    parser.add_argument('--trial_type', default='rsg')
    parser.add_argument('--rsg_intervals', nargs='*', type=int, default=None)
    parser.add_argument('--trial_args', nargs='*', help='terms to specify parameters of trial type')
    parser.add_argument('--trial_len', type=int, default=100)
    parser.add_argument('--n_trials', type=int, default=1000)
    args = parser.parse_args()

    if args.trial_args is None:
        args.trial_args = []

    if args.mode == 'create':
        dset = create_dataset(args)
        save_dataset(dset, args.name, args=args)
    elif args.mode == 'load':
        dset = load_dataset(args.name)

        dset_len = len(dset)
        sample = random.sample(dset, 6)
        dset_range = range(len(sample[0][0]))
        fig, ax = plt.subplots(2,3,sharex=True, sharey=True, figsize=(12,7))
        for i, ax in enumerate(fig.axes):
            ax.plot(dset_range, sample[i][0], color='coral', label='ready/set', lw=2)
            ax.plot(dset_range, sample[i][1], color='dodgerblue', label='go', lw=2)
            ax.set_title(sample[i][2])

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center')
        plt.show()
    # confirm ready set go works
    # for i in range(5):
    #     np.random.shuffle(dset)
    #     print(np.where(dset[i][0] == 1), np.argmax(dset[i][1]))
