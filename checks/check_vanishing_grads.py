import numpy as np
import torch

import matplotlib.pyplot as plt

import argparse
import pdb
import os
import sys

sys.path.append('../')

from utils import Bunch, load_rb
from train import Trainer, parse_args, adjust_args

args = parse_args()
args.no_log = True
args.optimizer = 'adam'
args.n_epochs = 3

args.dataset = '../' + args.dataset
args = adjust_args(args)

trainer = Trainer(args)

net = trainer.net

grad_norms = {
    'W_f.weight':[],
    'W_ro.weight':[],
    'W_ro.bias':[],
    'W_f.bias':[]
}

# takes in the same arguments that `iteration` spits out
def ix_cback(loss, etc):
    for k,v in net.named_parameters():
        if k in grad_norms.keys():
            grad_norms[k].append(torch.abs(v.grad).mean().item())

final_loss, n_iters = trainer.train(ix_callback=ix_cback)

pdb.set_trace()
