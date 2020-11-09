import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pdb
import os
import sys
sys.path.append('../')

from utils import get_config

csv_path = '../logs/3531677.csv'
# csv_path = '../logs/3529997.csv'
csv_data = pd.read_csv(csv_path)

vals = []
for i in csv_data.slurm_id:
    run_dir = os.path.join('../logs/3531677', str(i))
    run_files = os.listdir(run_dir)
    for f in run_files:
        if f.startswith('config'):
            c_file = os.path.join(run_dir, f)
    config = get_config(c_file, ctype='model')
    if config['train_parts'][0] == '':
        vals.append('all')
    else:
        vals.append('Wf-Wro')
csv_data['tparts'] = vals


color_scale = ['coral', 'chartreuse', 'skyblue']


cols_to_keep = ['slurm_id', 'N', 'D', 'seed', 'rseed', 'rnoise', 'dset', 'loss', 'tparts']
dt = csv_data[cols_to_keep]
dt = dt.sort_values(by='D')
dt.D = dt.D.astype(str)

rnoises = dt['rnoise'].unique()
rseeds = dt['rseed'].unique()

fig, axes = plt.subplots(nrows=len(rnoises), ncols=len(rseeds), sharex=True, sharey=True, figsize=(14,6))

# dt = dt[(dt.dset == 'datasets/rsg-sohn.pkl')]
for i, rnoise in enumerate(rnoises):
    for j, rseed in enumerate(rseeds):
        subset = dt[(dt.rnoise == rnoise) & (dt.rseed == rseed)]
        ax = axes[i, j]
        # ax.tick_params()
        # ax.xaxis.set_ticks_position('bottom')
        # ax.tick_params(which='major', width=1.00, length=4)
        ax.set_xlabel(rseed)
        ax.set_ylabel(rnoise)
        
        # subset_all = subset[subset.]
        subsubset = subset[subset.tparts == 'all']
        ax.scatter(subsubset.D, subsubset.loss, s=5, alpha=.7, c=color_scale[0], label='train all')
        subsubset = subset[subset.tparts == 'Wf-Wro']
        ax.scatter(subsubset.D, subsubset.loss, s=5, alpha=.7, c=color_scale[1], label='train parts')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axhline(y=0, color='dimgray', alpha = 1)
        # ax.tick_params(axis='both', color='white')
        ax.grid(None)
        ax.grid(True, which='major', axis='y', lw=1, color='lightgray', alpha=0.4)
        ax.set_xlim([-.5, 3.5])

plt.legend()
plt.show()
