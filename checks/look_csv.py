import pandas as pd
import numpy as np


import argparse
p = argparse.ArgumentParser()
p.add_argument('csv')
args = p.parse_args()

results = pd.read_csv(args.csv, quotechar='|')

results = results.sort_values(['tparts', 'rseed'])

del results['dset']

#results = results.sort_values('loss')

# print(results)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(results)