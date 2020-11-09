from itertools import product
import os
import json
import argparse
import random

def create_parameters(name):

    mapping = {}
    ix = 1

    Ds = [30, 100, 200]
    Ns = [100, 200]

    lr = 1e-4
    n_epochs = 20
    patience = 4000

    # keep the same network seeds
    preserve_seed = False

    n_seeds = 1
    n_rseeds = 3

    # noises = [0, 0.1, 0.01]
    noises = [0]
    train_parts = ['W_f', 'W_ro']

    debug = False
    if debug:
        Ns = [100]
        Ds = [30]
        n_seeds = 1
        n_rseeds = 1
        noises = [0]
        n_epochs = 5
        patience = 1000

    datasets = [
        'datasets/rsg-bin.pkl'
    ]
    losses = [
        'bce'
    ]

    if preserve_seed:
        seed_samples = random.sample(range(1000), n_seeds)

    rseed_samples = random.sample(range(1000), n_rseeds)

    for (nD, nN, d, seed, rseed, noise) in product(Ds, Ns, datasets, range(n_seeds), range(n_rseeds), noises):
        if nD > nN:
            continue
        run_params = {}
        run_params['dataset'] = d
        run_params['losses'] = losses
        run_params['D'] = nD
        run_params['N'] = nN

        # these parameters only useful when training with adam
        run_params['lr'] = lr
        run_params['n_epochs'] = n_epochs
        run_params['patience'] = patience

        # run_params['optimizer'] = 'lbfgs-scipy'
        run_params['optimizer'] = 'adam'

        run_params['train_parts'] = train_parts

        run_params['res_noise'] = noise

        # keep the seed the same across all runs sharing network seeds
        # but use a totally random one otherwise. train.py will take care of it
        if preserve_seed:
            run_params['seed'] = seed_samples[seed]
        run_params['res_seed'] = rseed_samples[rseed]

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
