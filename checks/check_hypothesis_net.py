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


from network import HypothesisNetwork

b = Bunch()
b.dataset = '../datasets/temp.pkl'
b.out_act = 'none'
b.L = 2
b.Z = 2

dset = load_rb(b.dataset)

net = HypothesisNetwork(b)


seq1 = dset[0]
t1 = seq1[0]

t1 = torch.Tensor(t1)

net(t1)
