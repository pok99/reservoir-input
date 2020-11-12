import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pdb
import os
import sys
import subprocess
sys.path.append('../')

from utils import get_config, load_rb
from testers import load_model_path, test_model


run_id = '3535712'
csv_path = f'../logs/{run_id}.csv'
csv_data = pd.read_csv(csv_path)


csv_data['tparts'].fillna('all', inplace=True)

cols_to_keep = ['slurm_id', 'N', 'D', 'seed', 'rseed', 'rnoise', 'mnoise', 'dset', 'loss', 'tparts']
dt = csv_data[cols_to_keep]

dsets = ['datasets/rsg-sohn-100-200.pkl']


dset_map = {
    'datasets/rsg-sohn-100-150.pkl': 'datasets/rsg-sohn-100-5-150.pkl',
    'datasets/rsg-sohn-100-200.pkl': 'datasets/rsg-sohn-100-9-200.pkl',
    'datasets/rsg-sohn-150-200.pkl': 'datasets/rsg-sohn-150-5-200.pkl',
}

# mapping Ds so we can plot it as factor later
dt = dt.sort_values(by=['D', 'mnoise', 'rseed'])
Ds = dt['D'].unique()
D_map = dict(zip(Ds, range(len(Ds))))
dt['D_map'] = dt['D'].map(D_map)

dt['D_map'] += np.random.normal(0, .05, len(dt['D_map']))

color_scale = ['orchid']

plt.figure(figsize=(5,4))
ax = plt.gca()
ax.plot(range(400), range(400), color='black', lw=.6, linestyle='--')

dt = dt[dt.D == 100]
dt = dt[dt.tparts == 'all']
dt = dt[(dt.seed == 12) & (dt.rseed > 0) & (dt.rnoise == 0.01)]

dt = dt[dt.mnoise == 6]

intervals = [{}, {}, {}]
for j, dset in enumerate(dsets):
    subset = dt[dt.dset == dset]

    for iterr in range(len(subset)):

        job_id = subset.iloc[iterr].slurm_id

        model_folder = os.path.join('..', 'logs', run_id, str(job_id))
        model_path = os.path.join(model_folder, 'model_best.pth')
        config = get_config(model_path, ctype='model', to_bunch=True)
        config.m_noise = 0
        config.dataset = dset_map[config.dataset]
        net = load_model_path(model_path, config=config)

        data, loss = test_model(net, config, n_tests=200, dset_base='../')
        dset = load_rb(os.path.join('..', config.dataset))

        distr = {}

        for k in range(len(data)):
            dset_idx, x, _, z, _ = data[k]
            r, s, g = dset[dset_idx][2]

            t_first = torch.nonzero(z >= 1)
            if len(t_first) > 0:
                t_first = t_first[0,0]
            else:
                t_first = len(x)

            val = np.array(t_first - s - 5)

            interval = g - s
            if interval not in distr:
                distr[interval] = [val]
            else:
                distr[interval].append(val)

        for k,v in distr.items():
            if k in intervals[j]:
                intervals[j][k] += v
            else:
                intervals[j][k] = v
    
    pts = []
    stats = []
    for k,v in intervals[j].items():
        for vv in v:
            pts.append((k,vv))
        stats.append([k, np.mean(v), np.std(v)])

    stats.sort(key=lambda x: x[0])

    pts = list(zip(*pts))
    stats = list(zip(*stats))
    pts = np.array(pts).astype(float)
    pts += np.random.normal(0, .8, size=pts.shape)
    ax.scatter(pts[0], pts[1], marker='o', color=color_scale[j], alpha=0.1, s=10)
    ax.errorbar(stats[0], stats[1], yerr=stats[2], marker='o', color=color_scale[j], mfc=color_scale[j], mec='black', ms=8, elinewidth=1)

    print('finished', j)



ax.set_xlim([90, 210])
ax.set_ylim([80, 220])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('target interval')
plt.ylabel('produced interval')
plt.yticks(fontsize=9)
plt.xticks(fontsize=9)
plt.grid(None)
plt.grid(True, which='major', axis='y', lw=1, color='lightgray', alpha=0.4)

plt.show()
