import numpy as np
import pickle
import os
import sys
import json
import pdb

import argparse

# toy ready set go dataset
def create_dataset(args):
    name = args.name
    t_type = args.trial_type
    n_trials = args.n_trials
    t_len = args.trial_len

    trials = []


    if t_type == 'rsg':

        for n in range(n_trials):
            trial_x = np.zeros((t_len))
            trial_y = np.zeros((t_len))
            # amount of time in between ready and set cues
            t_p = np.random.randint(2, t_len // 2 - 1)

            ready_time = np.random.randint(0, t_len - t_p * 2)
            set_time = ready_time + t_p
            go_time = set_time + t_p

            trial_x[ready_time] = 1
            trial_x[set_time] = 1
            trial_y[go_time] = 1

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
    parser.add_argument('--trial_len', type=int, default=100)
    parser.add_argument('--n_trials', type=int, default=1000)
    args = parser.parse_args()

    dset_name = 'rsg_short.pkl'

    if args.mode == 'create':
        dset = create_dataset(args)
        save_dataset(dset, args.name, args=args)
    elif args.mode == 'load':
        dset = load_dataset(args.name)

    # confirm ready set go works
    for i in range(5):
        np.random.shuffle(dset)
        print(np.where(dset[i][0] == 1), np.argmax(dset[i][1]))
