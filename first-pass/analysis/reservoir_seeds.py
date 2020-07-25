import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import json
import pdb
import re
import sys

# ids = ['2134766', '2134941']
# csv_paths = ['../logs/'+x+'.csv' for x in ids]

# cache_csv = 'cache/' + '_'.join(ids) + '.csv'

# if not os.path.exists(cache_csv):
#     datas = []
#     for i in ids:
#         csv_path = '../logs/'+i+'.csv'
#         folder_path = '../logs/'+i
#         extra_data = []
#         for slurm_id in os.listdir(folder_path):
#             # find config file for each job within the run
#             for file in os.scandir(os.path.join(folder_path, slurm_id)):
#                 if file.name.startswith('config'):
#                     json_path = file.path
#                     break
#             with open(json_path, 'r') as f:
#                 config = json.load(f)

#             # and then record the seed info
#             extra_data.append((int(slurm_id), config['seed'], config['reservoir_seed']))

#         # turn that seed info into useful csvs
#         df_extra = pd.DataFrame(extra_data, columns=['slurm_id', 'seed', 'reservoir_seed'])
#         csv_data = pd.read_csv(csv_path)

#         csv_data = csv_data.merge(df_extra, on='slurm_id')

#         datas.append(csv_data)

#     csv_data = pd.concat(datas)
#     csv_data.to_csv(cache_csv)

# else:
#     csv_data = pd.read_csv(cache_csv, index_col=0)

csv_path = '../logs/2676780.csv'
csv_data = pd.read_csv(csv_path)

csv_data['dset'] = csv_data['dset'].apply(lambda x: x.replace('.', '/').split('/')[1])
# csv_data = csv_data[csv_data['dset'] == 'rsg2']
csv_data = csv_data.sort_values(['rseed', 'N', 'D', 'loss'])

# csv_data = csv_data[csv_data.reservoir_seed == 0]

dsets = csv_data.dset.unique()

fig, ax = plt.subplots(nrows=3, ncols=6, sharex=True, figsize=(14,6))

color_scale = ['coral', 'chartreuse', 'skyblue']
colors = {}
ix = 0
for s in csv_data.rseed.unique():
    colors[s] = color_scale[ix]
    ix += 1

for i,d in enumerate(dsets):
    subset = csv_data[csv_data.dset == d]
    Ns = subset.N.unique()
    for j,n in enumerate(Ns):
        subset2 = subset[subset.N == n]
        for s in subset2.rseed.unique():
            subset3 = subset2[subset2.rseed == s]
            ax[j, i].scatter(subset3.D, subset3.loss, c=colors[s], marker='+', s=60, label=f'seed = {s}')
            ax[j, i].set_xticks(subset3.D.unique())
            ax[j, i].grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
            ax[j, i].spines['top'].set_visible(False)
            ax[j, i].spines['right'].set_visible(False)
            # need to include N too
            ax[j, i].set_ylabel('loss')
            ax[j, i].set_xlabel('D')
            ax[j, i].set_title(f'dset = {d}, N = {n}')

plt.legend()
plt.show()




# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(csv_data)