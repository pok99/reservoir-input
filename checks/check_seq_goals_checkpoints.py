import numpy as np
import torch

import matplotlib.pyplot as plt

import pickle
import pdb

path = '../logs/test4/checkpoints_6086684.pkl'


with open(path, 'rb') as f:
    checkpoints = pickle.load(f)

x = checkpoints[0]

pdb.set_trace()