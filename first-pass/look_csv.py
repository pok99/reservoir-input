import pandas as pd
import numpy as np


import argparse
p = argparse.ArgumentParser()
p.add_argument('id', default='')
args = p.parse_args()

results = pd.read_csv(f'logs/{args.id}.csv')

results = results[(results.N == 100) & (results.D == 10) & (results.seed == 920) & (results.rseed == 273)]
results = results.sort_values('dset')

#results = results.sort_values('loss')

#print(results)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(results)