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
csv_data = csv_data[csv_data['dset'].str.startswith('rsg2_') == False]
csv_data = csv_data.sort_values(['rseed', 'N', 'D', 'loss'])
csv_data = csv_data[csv_data['D'] != 5]


# csv_data = csv_data[csv_data.reservoir_seed == 0]

dsets = csv_data.dset.unique()

fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, figsize=(14,6))

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
            axis = ax[j, i]
            subset3 = subset2[subset2.rseed == s]
            axis.scatter(subset3.D, subset3.loss, c=colors[s], marker='+', s=60, label=f'seed = {s}')
            axis.set_xticks(subset3.D.unique())
            axis.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)
            axis.axvline(x=0, color='dimgray', alpha = 1)
            axis.axhline(y=0, color='dimgray', alpha = 1)
            axis.tick_params(axis='both', color='white')
            # need to include N too
            # axis.set_ylabel('loss')
            # axis.set_xlabel('D')
            axis.set_title(f'N = {n}')

fig.text(0.5, 0.04, 'D', ha='center', va='center')
fig.text(0.06, 0.5, 'loss', ha='center', va='center', rotation='vertical')

plt.legend()
plt.show()




# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(csv_data)