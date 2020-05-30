import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import random
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('run')

args = parser.parse_args()

run_path = os.path.join('logs', args.run)
folders = [f for f in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, f))]

for f in folders:
    csv_path = os.path.join(run_path, f, 'losses.csv')
    pd_data = pd.read_csv(csv_path)

    plt.plot(pd_data.ix, pd_data.avg_loss, lw=0.5)

plt.ylim([10, 15])

plt.show()
