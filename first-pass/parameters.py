from itertools import product
import os
import json
import argparse

def create_parameters(name):

    mapping = {}
    ix = 1

    Ds = [5, 10, 50, 100, 250]
    Ns = [50, 100, 250]

    lrs = [1e-3]
    n_epochs = 50
    patience = 4000

    n_seeds = 4
    n_rseeds = 2

    datasets = [
        'data/rsg_tl100_sc1.pkl',
        'data/rsg_tl100_sc1_s10.pkl',
        'data/rsg_tl100_sc1_s20.pkl',
        'data/rsg_tl100_sc1_s30.pkl',
        'data/rsg_tl100_sc1_s40.pkl',
    ]

    #n_commands = len(Ds) * len(Ns) * len(trial_lens) * len(singles) * len(lrs) * n_seeds

    for (nD, nN, d, seed, rseed) in product(Ds, Ns, datasets, range(n_seeds), range(n_rseeds)):
        if nD > nN:
            continue
        run_params = {}
        run_params['dataset'] = d
        run_params['D'] = nD
        run_params['N'] = nN
        # run_params['lr'] = lr
        # run_params['n_epochs'] = n_epochs
        # run_params['patience'] = patience
        run_params['optimizer'] = 'lbfgs-scipy'

        run_params['reservoir_seed'] = rseed

        mapping[ix] = run_params
        ix += 1

    n_commands = ix - 1

    fname = os.path.join('slurm_params', name + '.json')
    with open(fname, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f'Produced {n_commands} run commands in {fname}. Use with `sbatch --array=1-{n_commands} slurm_train.sbatch`.')

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
