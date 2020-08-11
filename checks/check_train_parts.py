import numpy as np
import torch

import matplotlib.pyplot as plt

import argparse
import pdb
import os
import sys

sys.path.append('../')

from network import Network
from utils import Bunch, load_rb
from train import Trainer, parse_args, adjust_args

args = parse_args()
args.no_log = True
args.maxiter = 70
args.dataset = '../' + args.dataset
args.train_parts = ['W_ro']
args = adjust_args(args)

trainer = Trainer(args)

net = trainer.net

Wf = net.W_f.weight.detach().clone().numpy()
Wro = net.W_ro.weight.detach().clone().numpy()
J = net.reservoir.J.weight.detach().clone().numpy()

beginning = [Wf, Wro, J]

final_loss, n_iters = trainer.optimize_lbfgs('scipy')


Wf = net.W_f.weight.detach().clone().numpy()
Wro = net.W_ro.weight.detach().clone().numpy()
J = net.reservoir.J.weight.detach().clone().numpy()

ending = [Wf, Wro, J]

print('W_f:', np.mean(np.abs(ending[0] - beginning[0])))
print('W_ro:', np.mean(np.abs(ending[1] - beginning[1])))
print('J:', np.mean(np.abs(ending[2] - beginning[2])))


pdb.set_trace()