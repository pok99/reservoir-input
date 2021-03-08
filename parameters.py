from itertools import product
import os
import json
import argparse
import random

def create_parameters(name):

    mapping = {}
    ix = 1

    Ds = [10, 50]
    Ns = [200]

    lr = 1e-4
    n_epochs = 20
    patience = 4000
    batch_size = 2
    l2 = 0.5

    # keep the same network seeds
    preserve_seed = True

    # usually have this off but if we wanna check models, set it on
    log_checkpoint_models = False

    n_seeds = 2
    n_rseeds = 1

    m_noises = [0, 2]
    r_noises = [0.01]
    train_parts = [[''], ['W_f', 'W_ro']]

    datasets = [
        'datasets/rsg-2c.pkl'
        # 'datasets/rsg-sohn-100-200.pkl',
        # 'datasets/rsg-sohn-150-200.pkl',
        # 'datasets/rsg-sohn-100-150.pkl',
        # 'datasets/rsg-sohn-50-100.pkl',
        # 'datasets/rsg-sohn-50-150.pkl',
        # 'datasets/rsg-sohn-50-200.pkl',
        # 'datasets/rsg-sohn.pkl'
        # 'datasets/copy-snip-200.pkl',
        # 'datasets/copy-delay-20.pkl'
    ]
    losses = [
        'mse', 'mse-w'
    ]

    # losses = [
    #     'mse'
    # ]

    debug = False
    if debug:
        Ns = [100]
        Ds = [30]
        n_seeds = 1
        n_rseeds = 1
        noises = [0]
        n_epochs = 5
        patience = 1000
        train_parts = [['W_f', 'W_ro']]

    if preserve_seed:
        seed_samples = random.sample(range(1000), n_seeds)

    rseed_samples = random.sample(range(1000), n_rseeds)

    # seed_samples = [811, 946, 122]
    # rseed_samples = [492, 496, 291, 127, 727]

    seed_samples = [11, 12]
    rseed_samples = [1, 2, 3]

    for (d, nN, nD, rnoise, mnoise, tp, seed, rseed) in product(datasets, Ns, Ds, r_noises, m_noises, train_parts, range(n_seeds), range(n_rseeds)):
        if nD > nN:
            continue
        run_params = {}
        run_params['dataset'] = d
        run_params['losses'] = losses
        # run_params['l2'] = l2
        run_params['D'] = nD
        run_params['N'] = nN

        # these parameters only useful when training with adam
        run_params['lr'] = lr
        run_params['n_epochs'] = n_epochs
        run_params['patience'] = patience
        run_params['batch_size'] = batch_size

        # run_params['optimizer'] = 'lbfgs-scipy'
        run_params['optimizer'] = 'adam'
        # run_params['s_rate'] = 0.2

        run_params['train_parts'] = tp

        run_params['res_noise'] = rnoise
        run_params['m_noise'] = mnoise

        run_params['same_signal'] = False

        # keep the seed the same across all runs sharing network seeds
        # but use a totally random one otherwise. train.py will take care of it
        if preserve_seed:
            run_params['seed'] = seed_samples[seed]
        run_params['res_seed'] = rseed_samples[rseed]

        run_params['log_checkpoint_models'] = log_checkpoint_models

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
