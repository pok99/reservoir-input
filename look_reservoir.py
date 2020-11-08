import numpy as np
import matplotlib.pyplot as plt

import random
import pickle
import pdb
import torch

import argparse
from utils import Bunch, load_rb, get_config

# for plotting some instances over the course of training

from testers import load_model_path
from network import Reservoir

# args = Bunch(N=250, D=100, O=1, res_init_type='gaussian', res_init_params={'std': 2}, reservoir_seed=0)

# net = Network(args)

# trial_len = 100
# t = np.arange(trial_len)

# ins = []
# outs = []
# inps = torch.rand((12)) * 2 - 1
# for i in range(12):
#     net.reset(res_state_seed=np.random.randint(30))

#     #inp = torch.normal(torch.zeros(trial_len), 500*torch.ones(trial_len))
#     inp = torch.zeros(trial_len) * inps[i]

#     out = []
#     for j in inp:
#         out.append(net(j)[0].detach().item())

#     ins.append(inp)
#     outs.append(out)


# fig, ax = plt.subplots(3,4,sharex=True, sharey=True, figsize=(12,7))

# for i, ax in enumerate(fig.axes):

#     ax.axvline(x=0, color='dimgray', alpha = 1)
#     ax.axhline(y=0, color='dimgray', alpha = 1)
#     ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)

#     ax.plot(t, ins[i], color='coral', alpha=0.5, lw=1, label='inp')
#     ax.plot(t, outs[i], color='cornflowerblue', alpha=1, lw=1.5, label='out')

#     ax.tick_params(axis='both', color='white')

#     ax.set_ylim([-2,2])

# fig.text(0.5, 0.04, 'timestep', ha='center', va='center')
# fig.text(0.06, 0.5, 'value', ha='center', va='center', rotation='vertical')

# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='center right')

# plt.show()

def main(args):
    config = get_config(args.model)
    net = load_model_path(args.model, config)

    t_len = 200
    
    with torch.no_grad():
        net.reset()

        ins = torch.zeros((12))

        for j in range(t_len):
            


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model')
    args = p.parse_args()

    

    main(args)