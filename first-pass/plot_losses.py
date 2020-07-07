import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import random
import pickle

import argparse

# for plotting the losses of all the jobs within a single run
# doesn't give a very nice graph
# generally don't use this

parser = argparse.ArgumentParser()
parser.add_argument('run')

args = parser.parse_args()

run_path = os.path.join('logs', args.run)
folders = [f for f in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, f))]

for f in folders:
    csv_path = os.path.join(run_path, f, 'losses.csv')
    print(csv_path)
    if os.path.isfile(csv_path):
        pd_data = pd.read_csv(csv_path)
        plt.plot(pd_data.ix, pd_data.avg_loss, lw=0.5)

plt.ylim([2, 5])

plt.show()
