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


run_id = '3540770' # run across many Ds

csv_path = f'../logs/{run_id}.csv'
csv_data1 = pd.read_csv(csv_path)

run_id = '3541548' # across even more Ds

csv_path = f'../logs/{run_id}.csv'
csv_data2 = pd.read_csv(csv_path)

csv_data = pd.concat([csv_data1, csv_data2])


csv_data['tparts'].fillna('all', inplace=True)

cols_to_keep = ['slurm_id', 'N', 'D', 'seed', 'rseed', 'rnoise', 'mnoise', 'dset', 'loss', 'tparts']
dt = csv_data[cols_to_keep]
dt = dt.sort_values(by=['D', 'mnoise', 'rseed'])


# mapping Ds so we can plot it as factor later
Ds = dt['D'].unique()
D_map = dict(zip(Ds, range(len(Ds))))
dt['D_map'] = dt['D'].map(D_map)

dt['D_map'] += np.random.normal(0, .05, len(dt['D_map']))

rnoises = dt['rnoise'].unique()
mnoises = dt['mnoise'].unique()
rseeds = dt['rseed'].unique()

color_scale = ['coral', 'cornflowerblue', 'skyblue']

dt = dt[(dt.mnoise == 4)]

plt.figure(figsize=(5,4))
ax = plt.gca()
ax.set_xticklabels([0] + list(Ds), fontsize=9)
plt.yticks(fontsize=9)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
train_all = dt[dt.tparts == 'all']
train_lim = dt[dt.tparts == 'W_f-W_ro']
plt.scatter(train_all.D_map, train_all.loss, s=8, alpha=.3, c=color_scale[0])
plt.scatter(train_lim.D_map, train_lim.loss, s=8, alpha=.3, c=color_scale[1])
q_all = []
q_lim = []
for D in Ds:
    try:
        # q = np.quantile(train_all[train_all.D == D]['loss'], [.25, .5, .75])
        # q = [q[1], q[1] - q[0], q[2] - q[1]]
        mn = np.mean(train_all[train_all.D == D]['loss'])
        sd = np.std(train_all[train_all.D == D]['loss'])
        q = [mn, sd, sd]
        q_all.append(q)
    except IndexError:
        q_all.append([0,0,0])
    try:
        # q = np.quantile(train_lim[train_lim.D == D]['loss'], [.25, .5, .75])
        # q = [q[1], q[1] - q[0], q[2] - q[1]]
        mn = np.mean(train_lim[train_lim.D == D]['loss'])
        sd = np.std(train_lim[train_lim.D == D]['loss'])
        q = [mn, sd, sd]
        q_lim.append(q)
    except IndexError:
        q_lim.append([0,0,0])
q_all = list(zip(*q_all))
q_lim = list(zip(*q_lim))
plt.errorbar(range(len(Ds)), q_all[0], yerr = [q_all[1], q_all[2]], c=color_scale[0], ms=20, elinewidth=2, lw=3, label='trained RNN')
plt.errorbar(range(len(Ds)), q_lim[0], yerr = [q_lim[1], q_lim[2]], c=color_scale[1], ms=20, elinewidth=2, lw=3, label='fixed RNN')

# plt.plot(range(len(Ds)), means, c=color_scale[1], ms=50)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(None)
plt.grid(True, which='major', axis='y', lw=1, color='lightgray', alpha=0.4)
plt.xlim([-.5, len(Ds) - .5])
plt.ylim([0, 70])

plt.xlabel('D')
plt.ylabel('mean squared error')

plt.legend()
plt.show()
