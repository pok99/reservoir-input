import os

#import tensorflow as tf

import yaml
import logging
import time
import json
import csv
# import pandas as pd


# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LogObject(object):
    pass

# use yaml config files; note what is actually set via the config file
def add_yaml_args(args, config_file):
    if config_file:
        config = yaml.safe_load(open(config_file))
        dic = vars(args)
        # all(map(dic.pop, config))
        for c, v in config.items():
            dic[c] = v
            # if c in dic.keys():
            #     logging.info(f'{c} is set via config: {v}')
            # else:
            #     logging.warning(f'{c} is not set to begin with: {v}')
    return args


# produce run id and create log directory
def log_this(config, log_dir, log_name=None, checkpoints=True, use_id=True):
    run_id = str(int(time.time() * 100))[-7:]
    print('\n=== Logging ===', flush=True)
    
    if log_name is None or len(log_name) == 0:
        log_name = run_id
    print(f'Run id: {run_id} with name {log_name}', flush=True)

    run_dir = os.path.join(log_dir, log_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f'Log folder: {run_dir}', flush=True)

    log_path = os.path.join(run_dir, f'log_{run_id}.log')
    print(f'Log file: {log_path}', flush=True)

    if checkpoints:
        checkpoint_dir = os.path.join(run_dir, f'checkpoints_{run_id}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f'Logging checkpoints to {checkpoint_dir}', flush=True)
    else:
        checkpoint_dir = None

    # might want to send stdout here later too
    path_config = os.path.join(run_dir, f'config_{run_id}.json')
    with open(path_config, 'w', encoding='utf-8') as f:
        json.dump(vars(config), f, indent=4)
        print(f'Config file saved to: {path_config}', flush=True)

    log = LogObject()
    log.checkpoint_dir = checkpoint_dir
    log.log_dir = run_dir
    log.run_log = log_path
    log.run_id = run_id

    
    print('===============\n', flush=True)
    return log


