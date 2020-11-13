

import os
import subprocess

import pandas as pd



# for extracting best loss from config and log files when csv doesn't happen
run_id = '3532397'
losses = []
c_dict = {}
Ns = []
for i in range(1, 241):
    folder = os.path.join('../logs', run_id, str(i))
    files = os.listdir(folder)
    for f in files:
        if f.startswith('config'):
            c_file = os.path.join(folder, f)
        if f.startswith('log'):
            l_file = os.path.join(folder, f)
    config = get_config(c_file, ctype='model')
    best_loss = float(subprocess.check_output(['tail', '-1', l_file])[:-1].decode('utf-8').split(' ')[-1])
    losses.append(best_loss)
    for k,v in config.items():
        if k == 'train_parts':
            if len(v[0]) == 0:
                v = 'all'
            else:
                v = 'Wf-Wro'
        if k not in c_dict:
            c_dict[k] = [v]
        else:
            c_dict[k].append(v)
dt = pd.DataFrame.from_dict(c_dict)
dt['loss'] = losses
dt = dt.rename(columns={'train_parts':'tparts', 'dataset':'dset', 'res_noise':'rnoise', 'res_seed':'rseed'})



# for replacing 'train_parts' array with strings
vals = []
for i in csv_data.slurm_id:
    run_dir = os.path.join('../logs/3531677', str(i))
    run_files = os.listdir(run_dir)
    for f in run_files:
        if f.startswith('config'):
            c_file = os.path.join(run_dir, f)
    config = get_config(c_file, ctype='model')
    if config['train_parts'][0] == '':
        vals.append('all')
    else:
        vals.append('Wf-Wro')
csv_data['tparts'] = vals



# mapping Ds so we can plot it as factor later
Ds = dt['D'].unique()
D_map = dict(zip(Ds, range(len(Ds))))
dt['D_map'] = dt['D'].map(D_map)


# combining csvs
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