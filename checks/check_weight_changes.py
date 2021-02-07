import numpy as np
import torch
import matplotlib.pyplot as plt

import os
import sys
import json
import pdb


sys.path.append('../')

from testers import load_model_path



def main():
    ckpt_path = '../logs/test_mid/checkpoints_0393265'
    config_path = '../logs/test_mid/config_0393265.json'

    models = os.scandir(ckpt_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    norms = []

    for ix, s in enumerate(models):
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
        print(f'finished {ix}')


    plt.plot(norms)
    plt.show()


if __name__ == '__main__':
    main()