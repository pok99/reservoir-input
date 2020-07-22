from itertools import product
import os
import json
import argparse
import random

def create_parameters(name):

    mapping = {}
    ix = 1

    Ds = [5, 10, 50]
    Ns = [50, 100]

    lr = 1e-4
    n_epochs = 30
    patience = 4000

    # keep the same network seeds
    preserve_seed = False

    n_seeds = 1
    n_rseeds = 2

    biases = [True]

    datasets = [
        'datasets/rsg2.pkl',
        'datasets/rsg2_25.pkl',
        'datasets/rsg2_65.pkl',
        'datasets/copy2.pkl',
        'datasets/copy2_d50.pkl',
        'datasets/copy2_d100.pkl'
    ]

    #n_commands = len(Ds) * len(Ns) * len(trial_lens) * len(singles) * len(lrs) * n_seeds

    if preserve_seed:
        seed_samples = random.sample(range(1000), n_seeds)

    rseed_samples = random.sample(range(1000), n_rseeds)

    for (nD, nN, d, seed, rseed, bias) in product(Ds, Ns, datasets, range(n_seeds), range(n_rseeds), biases):
        if nD > nN:
            continue
        run_params = {}
        run_params['dataset'] = d
        run_params['D'] = nD
        run_params['N'] = nN

        if bias:
            run_params['bias'] = True
        else:
            run_params['bias'] = False

        # these parameters only useful when training with adam
        run_params['lr'] = lr
        run_params['n_epochs'] = n_epochs
        run_params['patience'] = patience

        # run with lbfgs instead - it's better
        run_params['optimizer'] = 'adam'

        run_params['train_parts'] = ['W_ro', 'W_f', 'reservoir']

        # keep the seed the same across all runs sharing network seeds
        # but use a totally random one otherwise. train.py will take care of it
        if preserve_seed:
            run_params['seed'] = seed_samples[seed]
        run_params['reservoir_seed'] = rseed_samples[rseed]

        mapping[ix] = run_params
        ix += 1

    n_commands = ix - 1

    fname = os.path.join('slurm_params', name + '.json')
    with open(fname, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f'Produced {n_commands} run commands in {fname}. Use with:\nsbatch --array=1-{n_commands} slurm_train.sbatch')

    return mapping


def apply_parameters(filename, args):
    dic = vars(args)
    with open(filename, 'r') as f:
        mapping = json.load(f)
    for k,v in mapping[str(args.slurm_id)].items():
        dic[k] = v
    return args


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--name', type=str, default='params')
    args = p.parse_args()

    create_parameters(args.name)
