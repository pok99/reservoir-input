import numpy as np
import pickle
import os
import sys
import json

# toy ready set go dataset
def create_dataset(args):
    name = args['name']
    t_type = args['trial_type']
    n_trials = args['n_trials']
    t_len = args['trial_len']

    trials = []


    if t_type == 'ready-set-go':

        for n in range(n_trials):
            trial_x = np.zeros((t_len))
            trial_y = np.zeros((t_len))
            # amount of time in between ready and set cues
            t_p = np.random.randint(10, t_len // 2 - 1)

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
        'trial_len': 100
    }

    dset_name = 'rsg_1'

    mode = 'create'

    if mode == 'create':
        dset = create_dataset(DEFAULT_ARGS)
        save_dataset(dset, dset_name, args=DEFAULT_ARGS)
    elif mode == 'load':

        dset = load_dataset(os.path.join('data', dset_name))

    # confirm ready set go works
    for i in range(5):
        print(np.where(dset[i][0] == 1), np.argmax(dset[i][1]))
