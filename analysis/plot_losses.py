import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pdb
import os
import sys
import subprocess
sys.path.append('../')

from utils import get_config

csv_path = '../logs/3533296.csv'
csv_data = pd.read_csv(csv_path)

csv_data['tparts'].fillna('all', inplace=True)

cols_to_keep = ['slurm_id', 'N', 'D', 'seed', 'rseed', 'rnoise', 'dset', 'loss', 'tparts']
dt = csv_data[cols_to_keep]
dt = dt.sort_values(by=['D', 'rnoise', 'rseed'])


# mapping Ds so we can plot it as factor later
Ds = dt['D'].unique()
D_map = dict(zip(Ds, range(len(Ds))))
dt['D_map'] = dt['D'].map(D_map)

dt['D_map'] += np.random.normal(0, .05, len(dt['D_map']))

rnoises = dt['rnoise'].unique()
rseeds = dt['rseed'].unique()

color_scale = ['coral', 'chartreuse', 'skyblue']

fig, axes = plt.subplots(nrows=len(rnoises), ncols=len(rseeds), sharex=True, sharey=True, figsize=(14,6))
fig.text(0.07, 0.5, 'loss', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'D', ha='center')

# dt = dt[(dt.dset == 'datasets/rsg-sohn.pkl')]
# dt = dt[(dt.tparts == 'all')]
for i, rnoise in enumerate(rnoises):
    for j, rseed in enumerate(rseeds):
        subset = dt[(dt.rnoise == rnoise) & (dt.rseed == rseed)]

        ax = axes[i, j]
        ax.set_xticklabels([0, 10, 30, 100, 200])
        # ax.tick_params()
        # ax.xaxis.set_ticks_position('bottom')
        # ax.tick_params(which='major', width=1.00, length=4)
        if i == 0:
            ax.set_title('seed = ' + str(rseed))
        if j == 0:
            ax.set_ylabel('noise = ' + str(rnoise))
        
        # subset_all = subset[subset.]
        train_all = subset[subset.tparts == 'all']
        ax.scatter(train_all.D_map, train_all.loss, s=8, alpha=.7, c=color_scale[0], label='train all')
        means = []
        for D in Ds:
            means.append(np.mean(train_all[train_all.D == D]['loss']))
        ax.plot(range(len(Ds)), means, c=color_scale[0], ms=20)
        train_lim = subset[subset.tparts == 'Wf-Wro']
        ax.scatter(train_lim.D_map, train_lim.loss, s=8, alpha=.7, c=color_scale[1], label='train part')
        means = []
        for D in Ds:
            means.append(np.mean(train_lim[train_lim.D == D]['loss']))
        ax.plot(range(len(Ds)), means, c=color_scale[1], ms=50)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.axhline(y=0, color='dimgray', alpha = 1)
        # ax.axvline(x=-.5, color='dimgray', alpha = 1)
        # ax.tick_params(axis='both', color='white')
        ax.grid(None)
        ax.grid(True, which='major', axis='y', lw=1, color='lightgray', alpha=0.4)
        ax.set_xlim([-.5, 3.5])
        ax.set_ylim([0, 100])


plt.legend()
plt.show()
