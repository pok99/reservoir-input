import numpy as np
from scipy.stats import norm
import pickle
import os
import sys
import json
import pdb

import matplotlib.pyplot as plt

import argparse

# toy ready set go dataset
def create_dataset(args):
    name = args.name
    t_type = args.trial_type
    n_trials = args.n_trials
    t_len = args.trial_len
    trial_args = args.trial_args

    trials = []

    if t_type == 'rsg':

        # check if we just want one interval in entire dataset
        if 'single' in trial_args:
            single_idx = trial_args.index('single')
            single_num = int(trial_args[single_idx + 1])
            assert single_num < t_len / 2
            t_p = np.random.randint(single_num, t_len - single_num)

        for n in range(n_trials):

            if 'single' not in trial_args:
                # amount of time in between ready and set cues
                t_p = np.random.randint(2, t_len // 2 - 1)

            ready_time = np.random.randint(0, t_len - t_p * 2)
            set_time = ready_time + t_p
            go_time = set_time + t_p

            # output 0s and 1s instead of pdf, use with CrossEntropyLoss
            if 'delta' in trial_args:
                trial_x = np.zeros((t_len))
                trial_y = np.zeros((t_len))
                
                trial_x[ready_time] = 1
                trial_x[set_time] = 1
                trial_y[go_time] = 1
            else:
                # check if width of gaussian is changed from default
                if 'scale' in trial_args:
                    scale_idx = trial_args.index('scale')
                    scale = trial_args[scale_idx + 1]
                else:
                    scale = 2

                trial_range = np.arange(t_len)
                trial_x = norm.pdf(trial_range, loc=ready_time, scale=scale)
                trial_x += norm.pdf(trial_range, loc=set_time, scale=scale)
                trial_y = 4 * norm.pdf(trial_range, loc=go_time, scale=scale)


            trials.append((trial_x, trial_y))

    return trials

def save_dataset(dset, name, args=None):
    fname = name + '.pkl'
    with open(os.path.join('data', fname), 'wb') as f:
        pickle.dump(dset, f)
    gname = name + '.json'
    if args is not None:
        args = vars(args)
        with open(os.path.join('data', gname), 'w') as f:
            json.dump(args, f)

def load_dataset(fpath):
    with open(fpath, 'rb') as f:
        dset = pickle.load(f)
    return dset


if __name__ == '__main__':

    DEFAULT_ARGS = {
        'name': 'default',
        'trial_type': 'ready-set-go',
        'n_trials': 1000,
        'trial_len': 40
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='load')
    parser.add_argument('name')
    parser.add_argument('--trial_type', default='rsg')
    parser.add_argument('--trial_args', nargs='*', help='terms to specify parameters of trial type')
    parser.add_argument('--trial_len', type=int, default=100)
    parser.add_argument('--n_trials', type=int, default=1000)
    args = parser.parse_args()

    if args.mode == 'create':
        dset = create_dataset(args)
        save_dataset(dset, args.name, args=args)
    elif args.mode == 'load':
        dset = load_dataset(args.name)

    # confirm ready set go works
    # for i in range(5):
    #     np.random.shuffle(dset)
    #     print(np.where(dset[i][0] == 1), np.argmax(dset[i][1]))
