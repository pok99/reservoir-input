import pandas as pd
import numpy as np


import argparse
p = argparse.ArgumentParser()
p.add_argument('id', default='')
args = p.parse_args()

results = pd.read_csv(f'logs/{args.id}.csv')

results = results.sort_values('loss')

print(results)