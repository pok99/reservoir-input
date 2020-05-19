import numpy as np
import matplotlib.pyplot as plt

import pickle


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('file')

args = parser.parse_args()

with open(args.file, 'rb') as f:
    data = pickle.load(f)

interval = len(data) // 12
data_intervals = [0]
for i in range(11):
    data_intervals.append(data_intervals[-1] + interval)

data = [data[i] for i in data_intervals]

fig, ax = plt.subplots(3,4,sharex=True, sharey=True, figsize=(12,7))

for i, ax in enumerate(fig.axes):
    ix, x, y, z, total_loss, avg_loss = data[i]
    xr = np.arange(len(x))

    ax.axvline(x=0, color='dimgray', alpha = 1)
    ax.axhline(y=0, color='dimgray', alpha = 1)
    ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.plot(xr, x, color='coral', alpha=0.5, lw=1, label='ready/set')
    ax.plot(xr, y, color='coral', alpha=1, lw=1, label='go')
    ax.plot(xr, z, color='cornflowerblue', alpha=1, lw=1.5, label='response')

    ax.tick_params(axis='both', color='white')

    ax.set_title(f'step {ix}, avg loss {round(avg_loss, 1)}', size='small')

fig.text(0.5, 0.04, 'timestep', ha='center', va='center')
fig.text(0.06, 0.5, 'value', ha='center', va='center', rotation='vertical')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')

plt.show()
