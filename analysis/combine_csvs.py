
import numpy as np
import pandas as pd
import os


ids = ['2134766', '2134941']
csv_paths = ['../logs/'+x+'.csv' for x in ids]

cache_csv = 'cache/' + '_'.join(ids) + '.csv'


datas = []
for i in ids:
    csv_path = '../logs/'+i+'.csv'
    folder_path = '../logs/'+i

    # turn that seed info into useful csvs
    df_extra = pd.DataFrame(extra_data, columns=['slurm_id', 'seed', 'reservoir_seed'])
    csv_data = pd.read_csv(csv_path)

    csv_data = csv_data.merge(df_extra, on='slurm_id')

    datas.append(csv_data)

csv_data = pd.concat(datas)
csv_data.to_csv(cache_csv)