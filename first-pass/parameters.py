from itertools import product
import os
import json
import argparse

def create_parameters(name):

    mapping = {}
    ix = 1

    Ds = [10, 20, 50, 100]
    Ns = [100]

    trial_lens = [100]
    singles = [10, 20, 30]
    lrs = [1e-2, 1e-3]
    n_epochs = 20

    n_commands = len(Ds) * len(Ns) * len(trial_lens) * len(singles) * len(lrs)

    for (nD, nN, tl, s, lr) in product(Ds, Ns, trial_lens, singles, lrs):
        run_params = {}
        run_params['dataset'] = f'data/rsg_tl100_s{s}.pkl'
        run_params['D'] = nD
        run_params['N'] = nN
        run_params['lr'] = lr
        run_params['n_epochs'] = n_epochs

        mapping[ix] = run_params
        ix += 1

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