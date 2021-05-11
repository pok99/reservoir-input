import numpy as np
import torch

import argparse
import os
import pdb

from testers import load_model_path

def main(args):
    model_folder = os.path.join(*args.model.split('/')[:-1])
    net = load_model_path(args.model)

    Mu = net.M_u.weight.data

    n_stim_dims = 3
    Mu = torch.roll(Mu, n_stim_dims, dims=1)

    net.M_u.weight.data = Mu

    torch.save(net.state_dict(), os.path.join(model_folder, 'model_test.pth'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model', type=str, help='model file to dissect')
    args = ap.parse_args()
    main(args)