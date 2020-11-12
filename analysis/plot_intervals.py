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

dsets =dt.dset.unique()


# mapping Ds so we can plot it as factor later
dt = dt.sort_values(by=['D', 'mnoise', 'rseed'])
Ds = dt['D'].unique()
D_map = dict(zip(Ds, range(len(Ds))))
dt['D_map'] = dt['D'].map(D_map)

dt['D_map'] += np.random.normal(0, .05, len(dt['D_map']))

# rnoises = dt['rnoise'].unique()
mnoises = dt['mnoise'].unique()

color_scale = ['coral', 'chartreuse', 'skyblue']

fig, axes = plt.subplots(nrows=len(mnoises), ncols=len(dsets), figsize=(14,10), squeeze=False)
fig.text(0.07, 0.5, 'loss', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'D', ha='center')

dt = dt[dt.D == 100]
dt = dt[dt.tparts == 'W_f-W_ro']
dt = dt[(dt.seed == 12) & (dt.rseed > 1) & (dt.rnoise == 0.01)]

# dt = dt[(dt.dset == 'datasets/rsg-sohn.pkl')]
# dt = dt[(dt.rnoise == 0.01)]
for i, mnoise in enumerate(mnoises):
    for j, dset in enumerate(dsets):
        subset = dt[(dt.mnoise == mnoise) & (dt.dset == dset)]
        ax = axes[i, j]

        for iterr in range(len(subset)):

            job_id = subset.iloc[iterr].slurm_id

            model_folder = os.path.join('..', 'logs', run_id, str(job_id))
            model_path = os.path.join(model_folder, 'model_best.pth')
            # for k in os.listdir(model_folder):
            #     if k.startswith('model_') and k != 'model_best.pth':
            #         model_path = os.path.join(model_folder, k)
            #         break
            config = get_config(model_path, ctype='model', to_bunch=True)
            config.m_noise = 0
            net = load_model_path(model_path, config=config)

            data, loss = test_model(net, config, n_tests=250, dset_base='../')
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

                mode = 'intervals'
                if mode == 'offsets':
                    val = t_first - g
                elif mode == 'times':
                    val = t_first
                elif mode == 'intervals':
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

            ax.scatter(intervals, vals, marker='o', color=color_scale[iterr], alpha=0.5)

        x_min, x_max = min(intervals), max(intervals)
        y_min, y_max = min(vals), max(vals)
        xdiff = x_max - x_min
        ydiff = y_max - y_min
        x_min -= .1 * xdiff; y_min -= .1 * ydiff
        x_max += .1 * xdiff; y_max += .1 * ydiff
        ax.plot(range(int(x_max)), range(int(x_max)))
        # plt.fill_between(intervals, offsets - stds, offsets, color='coral', alpha=.5)
        # plt.fill_between(intervals, offsets + stds, offsets, color='coral', alpha=.5)
        ax.set_xlabel('real t_p')

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        print('finished', i, j)

        # ax = axes[i, j]
        # ax.set_xticklabels([0] + list(Ds))
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # # ax.tick_params()
        # # ax.xaxis.set_ticks_position('bottom')
        # # ax.tick_params(which='major', width=1.00, length=4)
        # if i == 0:
        #     ax.set_title(dset)
        # if j == 0:
        #     ax.set_ylabel('noise = ' + str(mnoise))
        
        # # subset_all = subset[subset.]
        # train_all = subset[subset.tparts == 'all']
        # ax.scatter(train_all.D_map, train_all.loss, s=8, alpha=.7, c=color_scale[0], label='train all')
        # means = []
        # for D in Ds:
        #     means.append(np.mean(train_all[train_all.D == D]['loss']))
        # ax.plot(range(len(Ds)), means, c=color_scale[0], ms=20)
        # train_lim = subset[subset.tparts == 'W_f-W_ro']
        # ax.scatter(train_lim.D_map, train_lim.loss, s=8, alpha=.7, c=color_scale[1], label='train part')
        # means = []
        # for D in Ds:
        #     means.append(np.mean(train_lim[train_lim.D == D]['loss']))
        # ax.plot(range(len(Ds)), means, c=color_scale[1], ms=50)

        # train_ro = subset[subset.tparts == 'W_ro']
        # ax.scatter(train_ro.D_map, train_ro.loss, s=8, alpha=.7, c=color_scale[2], label='train ro')
        # means = []
        # for D in Ds:
        #     means.append(np.mean(train_ro[train_ro.D == D]['loss']))
        # ax.plot(range(len(Ds)), means, c=color_scale[2], ms=20)

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # # ax.spines['bottom'].set_visible(False)
        # # ax.spines['left'].set_visible(False)
        # # ax.axhline(y=0, color='dimgray', alpha = 1)
        # # ax.axvline(x=-.5, color='dimgray', alpha = 1)
        # # ax.tick_params(axis='both', color='white')
        # ax.grid(None)
        # ax.grid(True, which='major', axis='y', lw=1, color='lightgray', alpha=0.4)
        # ax.set_xlim([-.5, len(Ds) - .5])
        # ax.set_ylim([0, 20])


plt.legend()
plt.show()
