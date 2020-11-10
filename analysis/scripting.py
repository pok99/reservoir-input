

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