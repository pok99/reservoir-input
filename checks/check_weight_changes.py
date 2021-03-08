import numpy as np
import torch
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import os
import sys
import json
import pdb

import argparse

sys.path.append('../')
from testers import load_model_path

def main(args):

    files = os.scandir(args.dir)
    for fn in os.scandir(args.dir):
        if os.path.basename(fn).startswith('checkpoints'):
            ckpt_folder = fn
        elif os.path.basename(fn).startswith('config'):
            config_path = fn

    models = os.scandir(ckpt_folder)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    norms = []

    for ix, s in enumerate(models):
        print(s.path)
        model = load_model_path(s.path, config)
        # J = model.reservoir.J.weight.data.numpy()
        Wf = model.W_f.weight.data.numpy()

        if ix == 0:
            # last_J = J
            last_Wf = Wf
            continue

        # dif = J - last_J
        dif = Wf - last_Wf
        norms.append(np.linalg.norm(dif))

        # last_J = J
        last_Wf = Wf


    plt.plot(norms)
    plt.savefig('figures/weight_changes.png')


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('dir', type=str)
    args = ap.parse_args()


    main(args)
