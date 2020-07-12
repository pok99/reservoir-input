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
parser.add_argument('file', help='model file')
parser.add_argument('dataset', help='dataset we want to test it on')
args = parser.parse_args()

with open(args.file, 'rb') as f:
    model = torch.load(f)

dset = load_dataset(args.dataset)

data = test_model(model, dset, 500)

distr = {}

for i in range(len(data)):

    dset_idx, x, y, z, _ = data[i]
    r, s, g = dset[dset_idx][2]

    peak = np.argmax(z)
    dif = peak - g
    if np.abs(dif) > 20:
        continue

    interval = s - r
    if interval not in distr:
        distr[interval] = [dif]
    else:
        distr[interval].append(dif)

intervals = []
for k,v in distr.items():
    v_avg = np.mean(v)
    v_std = np.std(v)
    intervals.append((k,v_avg, v_std))

intervals.sort(key=lambda x: x[0])
intervals, offsets, stds = list(zip(*intervals))
offsets = np.array(offsets)
stds = np.array(stds)

plt.scatter(intervals, offsets, marker='o', color='tomato', alpha=0.5)
# plt.fill_between(intervals, offsets - stds, offsets, color='coral', alpha=.5)
# plt.fill_between(intervals, offsets + stds, offsets, color='coral', alpha=.5)
plt.xlabel('interval length')
plt.ylabel('average peak offset')

plt.xlim([0,100])


plt.show()