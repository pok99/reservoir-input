import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import random
import pickle
import argparse
import pdb
import sys

sys.path.append('../')

from helpers import test_model

from dataset import load_dataset
from reservoir import Network, Reservoir

parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('dataset')
args = parser.parse_args()

with open(args.file, 'rb') as f:
    model = torch.load(f)

dset = load_dataset(args.dataset)

net = nn.Linear(5, 5)
pdb.set_trace()


data = test_model(model, dset, 0)
pdb.set_trace()