import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

import argparse
import pdb
import os


parser = argparse.ArgumentParser()
parser.add_argument('csv')

args = parser.parse_args()


csv_data = pd.read_csv(args.csv, index_col=0)


dsets = ['datasets/copy_d10.pkl', 'datasets/copy_d20.pkl', 'datasets/copy_d30.pkl']
i = 0
for d in dsets:
    i += 10
    dset_filtered = csv_data[csv_data['dset'] == d]
    plt.scatter(np.random.random(len(dset_filtered)) + i - 0.5, dset_filtered['loss'], alpha=0.7)

plt.ylim([0, 15])
plt.xlabel('delay')
plt.ylabel('loss')
plt.show()