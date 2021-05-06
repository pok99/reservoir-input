import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.optimize import minimize

import os
import argparse
import pdb
import sys
import pickle
import logging
import random
import csv
import math
import json
import copy
import pandas as pd


from utils import log_this, load_rb, get_config, fill_args, update_args, get_file_args
from helpers import get_optimizer, get_scheduler, get_criteria, create_loaders

from tasks import *

from trainer import Trainer

# TODO
network_params = [
    'L', 'D1', 'D2', 'N', 'Z', 'net', 'x_noise', 'm_noise', 'res_noise', 'res_init_g',
    'train_parts', 'model_path', 'model_config_path', 
    'separate_signal', 'same_test',
    'optimizer', 'k', 'batch_size', 'lr', 'n_epochs', 'conv_type', 'patience', 'l2_reg', 's_rate', 'loss', 'l1', 'l2',
    'seed', 'res_seed', 'res_x_seed', 'res_burn_steps', 'res_x_init'
]


def parse_args():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-L', type=int, default=5, help='latent input dimension')
    parser.add_argument('--D1', type=int, default=50, help='u dimension')
    parser.add_argument('--D2', type=int, default=50, help='v dimension')
    parser.add_argument('-N', type=int, default=300, help='number of neurons in reservoir')
    # parser.add_argument('-Z', type=int, default=5, help='output dimension')

    parser.add_argument('--net', type=str, default='M2', choices=['basic', 'M2'])

    parser.add_argument('--train_parts', type=str, nargs='+', default=['M_u', 'M_ro'])
    parser.add_argument('--share_parts', type=str, nargs='+', default=['M_ro'])

    parser.add_argument('-c', '--config', type=str, default=None, help='use args from config file')
    
    # make sure model_config path is specified if you use any paths! it ensures correct dimensions, bias, etc.
    parser.add_argument('--model_config_path', type=str, default=None, help='config path corresponding to model load path')
    parser.add_argument('--model_path', type=str, default=None, help='start training from certain model. superseded by below')
    # parser.add_argument('--Wro_path', type=str, default=None, help='start training from certain Wro')
    # parser.add_argument('--Wf_path', type=str, default=None, help='start training from certain Wf')
    # parser.add_argument('--J_path', type=str, default=None, help='saved reservoir. should be saved with seed tho')
    
    # network manipulation
    # parser.add_argument('--res_init_type', type=str, default='gaussian', help='')
    parser.add_argument('--res_init_g', type=float, default=1.5)
    parser.add_argument('--res_noise', type=float, default=0)
    parser.add_argument('--x_noise', type=float, default=0)
    parser.add_argument('--m_noise', type=float, default=0)
    parser.add_argument('--no_bias', action='store_true')
    parser.add_argument('--m1_act', type=str, default='none', help='act fn bw M_f and W_f')
    parser.add_argument('--m2_act', type=str, default='none', help='act fn bw W_ro and M_ro')
    parser.add_argument('--out_act', type=str, default=None, help='output activation')

    parser.add_argument('-d', '--dataset', type=str, nargs='+', help='dataset(s) to use. >1 means different contexts')
    # high-level arguments that control dataset manipulations
    parser.add_argument('--separate_signal', action='store_true', help='use 2d input instead of combined ready/set pulses')
    parser.add_argument('--same_test', action='store_true', help='use entire dataset for both training and testing')
    
    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'rmsprop', 'lbfgs'], default='adam')
    parser.add_argument('--k', type=int, default=0, help='k for t-bptt. use 0 for full bptt')

    # adam parameters
    parser.add_argument('--batch_size', type=int, default=1, help='size of minibatch used')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. adam only')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train for. adam only')
    parser.add_argument('--conv_type', type=str, choices=['patience', 'grad'], default='patience', help='how to determine convergence. adam only')
    parser.add_argument('--patience', type=int, default=4000, help='stop training if loss doesn\'t decrease. adam only')
    parser.add_argument('--l2_reg', type=float, default=0, help='amount of l2 regularization')
    parser.add_argument('--s_rate', default=None, type=float, help='scheduler rate. dont use for no scheduler')
    parser.add_argument('--loss', type=str, nargs='+', default=['mse'])

    # adam lambdas
    parser.add_argument('--l1', type=float, default=1, help='weight of normal loss')
    parser.add_argument('--l2', type=float, default=1, help='weight of exponential loss')

    # lbfgs-scipy parameters
    parser.add_argument('--maxiter', type=int, default=10000, help='limit to # iterations. lbfgs-scipy only')

    # seeds
    parser.add_argument('--seed', type=int, help='general purpose seed')
    parser.add_argument('--network_seed', type=int, help='seed for network initialization')
    parser.add_argument('--res_seed', type=int, help='seed for reservoir')
    parser.add_argument('--res_x_seed', type=int, default=0, help='seed for reservoir init hidden states. -1 for zero init')
    parser.add_argument('--res_burn_steps', type=int, default=200, help='number of steps for reservoir to burn in')

    parser.add_argument('-x', '--res_x_init', type=str, default=None, help='other seed options for reservoir')

    # control logging
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--log_checkpoint_models', action='store_true')
    parser.add_argument('--log_checkpoint_samples', action='store_true')

    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--slurm_param_path', type=str, default=None)
    parser.add_argument('--slurm_id', type=int, default=None)
    parser.add_argument('--use_cuda', action='store_true')

    args = parser.parse_args()
    return args

def adjust_args(args):
    # don't use logging.info before we initialize the logger!! or else stuff is gonna fail

    # dealing with slurm. do this first!! before anything else
    # needs to be before seed setting, so we can set it
    if args.slurm_id is not None:
        from parameters import apply_parameters
        args = apply_parameters(args.slurm_param_path, args)

    # loading from a config file
    if args.config is not None:
        config = get_file_args(args.config)
        args = update_args(args, config)

    # setting seeds
    if args.res_seed is None:
        args.res_seed = random.randrange(1e6)
    if args.seed is None:
        args.seed = random.randrange(1e6)
    if args.network_seed is None:
        args.network_seed = random.randrange(1e6)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # TODO
    # in case we are loading from a model
    # if we don't use this we might end up with an error when loading model
    # uses a new seed
    if args.model_path is not None:
        config = get_config(args.model_path)
        args = fill_args(args, config, overwrite_none=True)
        enforce_same = ['N', 'D1', 'D2', 'net', 'bias', 'use_reservoir']
        for v in enforce_same:
            if v in config and args.__dict__[v] != config[v]:
                print(f'Warning: based on config, changed {v} from {args.__dict__[v]} -> {config[v]}')
                args.__dict__[v] = config[v]

    # shortcut for specifying train everything including reservoir
    if args.train_parts == ['all']:
        args.train_parts = ['']

    # TODO
    args.bias = not args.no_bias
    if 'rsg' in args.dataset[0]:
        args.out_act = 'exp'
    else:
        args.out_act = 'none'
    args.T = len(args.dataset)

    # initializing logging
    # do this last, because we will be logging previous parameters into the config file
    if not args.no_log:
        if args.slurm_id is not None:
            log = log_this(args, 'logs', os.path.join(args.name.split('_')[0], args.name.split('_')[1]), checkpoints=args.log_checkpoint_models)
        else:
            log = log_this(args, 'logs', args.name, checkpoints=args.log_checkpoint_models)

        logging.basicConfig(format='%(message)s', filename=log.run_log, level=logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(console)
        args.log = log
    else:
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        logging.info('NOT LOGGING THIS RUN.')

    # logging, when loading models from paths
    if args.model_path is not None:
        logging.info(f'Using model path {args.model_path}')
        if args.model_config_path is not None:
            logging.info(f'...with config file {args.model_config_path}')
        else:
            logging.info('...but not using any config file. Errors may ensue due to net param mismatches')

    return args


if __name__ == '__main__':
    args = parse_args()
    args = adjust_args(args)

    trainer = Trainer(args)
    logging.info(f'Initialized trainer. Using device {trainer.device}, optimizer {args.optimizer}.')

    if args.optimizer == 'lbfgs':
        best_loss, n_iters = trainer.optimize_lbfgs()
    elif args.optimizer in ['sgd', 'rmsprop', 'adam']:
        best_loss, n_iters = trainer.train()

    if args.slurm_id is not None:
        # if running many jobs, then we gonna put the results into a csv
        csv_path = os.path.join('logs', args.name.split('_')[0] + '.csv')
        csv_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            labels_csv = ['slurm_id', 'N', 'D1', 'D2', 'seed', 'rseed', 'mnoise', 'rnoise', 'dset', 'niter', 'tparts', 'loss']
            vals_csv = [
                args.slurm_id, args.N, args.D1, args.D2, args.seed,
                args.res_seed, args.m_noise, args.res_noise,
                args.dataset, n_iters, '-'.join(args.train_parts), best_loss
            ]
            if args.optimizer != 'lbfgs':
                labels_csv.extend(['lr', 'epochs'])
                vals_csv.extend([args.lr, args.n_epochs])

            if not csv_exists:
                writer.writerow(labels_csv)
            writer.writerow(vals_csv)

    logging.shutdown()

