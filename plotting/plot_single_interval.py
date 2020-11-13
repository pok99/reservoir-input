import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import random
import pickle
import argparse
import pdb
import json
import sys
import os

sys.path.append('../')

from utils import get_config, fill_undefined_args, load_rb
from testers import load_model_path, test_model

parser = argparse.ArgumentParser()
parser.add_argument('model', help='model file')
parser.add_argument('--mode', default='intervals', choices=['offsets', 'times', 'intervals'])
parser.add_argument('-c', '--config', default=None, type=str)
parser.add_argument('--n_samples', default=500, type=int)
args = parser.parse_args()


if args.config is None:
    config = get_config(args.model, ctype='model')
else:
    config = json.load(open(args.config, 'r'))
config = fill_undefined_args(args, config, overwrite_none=True)

net = load_model_path(args.model, config=config)
data, loss = test_model(net, config, n_tests=500, dset_base='../')
dset = load_rb(os.path.join('..', config.dataset))

distr = {}

for i in range(len(data)):

    dset_idx, x, _, z, _ = data[i]
    r, s, g = dset[dset_idx][2]

    t_first = torch.nonzero(z >= 1)
    if len(t_first) > 0:
        t_first = t_first[0,0]
    else:
        t_first = len(x)

    if args.mode == 'offsets':
        val = t_first - g
    elif args.mode == 'times':
        val = t_first
    elif args.mode == 'intervals':
        val = t_first - s

    val = np.asarray(val)

    interval = g - s + 5
    if interval not in distr:
        distr[interval] = [val]
    else:
        distr[interval].append(val)

intervals = []
for k,v in distr.items():
    v_avg = np.mean(v)
    v_std = np.std(v)
    intervals.append((k,v_avg, v_std))

intervals.sort(key=lambda x: x[0])
intervals, vals, stds = list(zip(*intervals))
vals = np.array(vals)
stds = np.array(stds)

plt.plot()
plt.scatter(intervals, vals, marker='o', color='tomato', alpha=0.5)

x_min, x_max = min(intervals), max(intervals)
y_min, y_max = min(vals), max(vals)
xdiff = x_max - x_min
ydiff = y_max - y_min
x_min -= .1 * xdiff; y_min -= .1 * ydiff
x_max += .1 * xdiff; y_max += .1 * ydiff
plt.plot(range(int(x_max)), range(int(x_max)))
# plt.fill_between(intervals, offsets - stds, offsets, color='coral', alpha=.5)
# plt.fill_between(intervals, offsets + stds, offsets, color='coral', alpha=.5)
plt.xlabel('real t_p')
label_str = ''
if args.mode == 'intervals':
    label_str = 't_p'
elif args.mode == 'offsets':
    label_str = 'offsets'
elif args.mode == 'times':
    label_str = 'times'
plt.ylabel('network ' + label_str)

plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])


plt.show()