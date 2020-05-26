import pandas as pd
import numpy as np



results = pd.read_csv('logs/2103141.csv')

results = results.sort_values('loss')

print(results)