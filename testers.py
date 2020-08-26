import numpy as np
import torch
import torch.nn as nn

import random
import os
import pdb
import json

from network import BasicNetwork, StateNet
from utils import Bunch, load_rb



# extracts the correct parameters N, D, O, etc. in order to properly create a net to load into
# TODO: load from config path instead it will be easier than wrestling with bugs
def load_model_path(path, params={}):
    path_folder = '/'.join(path.split('/')[:-1])
    model_id = path.split('_')[-1][:-4]
    config_path = os.path.join(path_folder, f'config_{model_id}.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    m_dict = torch.load(path)
    bunch = Bunch()
    bunch.N = config['N']
    bunch.D = config['D']
    bunch.L = config['L']
    bunch.Z = config['Z']
    bunch.T = config['T']
    bunch.bias = True

    #bunch.reservoir_burn_steps = 200
    bunch.reservoir_x_seed = 0
    #bunch.network_delay = 0

    #bunch.res_init_type = 'gaussian'
    #bunch.res_init_params = {'std': 1.5}
    #bunch.reservoir_seed = 0

    #bunch.reservoir_noise = 0
    if 'reservoir_noise' in params:
        bunch.reservoir_noise = params['reservoir_noise']

    #bunch.out_act = 'exp'
    if 'out_act' in params and params['out_act'] is not None:
        bunch.out_act = params['out_act']
    elif 'dset' in params:
        if 'rsg' in params['dset']:
            bunch.out_act = 'exp'
        else:
            bunch.out_act = 'none'

    if 'stride' in params and params['stride'] is not None:
        bunch.stride = params['stride']

    # bunch.bias = True
    # if 'W_f.bias' not in m_dict:
    #     bunch.bias = False
    #     # m_dict['W_f.bias'] = torch.zeros(bunch.D)
    #     # m_dict['W_ro.bias'] = torch.zeros(bunch.Z)

    if config['net'] == 'basic':
        net = BasicNetwork(bunch)
    elif config['net'] == 'state':
        net = StateNet(bunch)
    net.load_state_dict(m_dict)
    net.eval()


    return net

# given a model and a dataset, see how well the model does on it
# works with plot_trained.py
def test_model(net, dset, n_tests=0):

    criterion = nn.MSELoss()
    dset_idx = range(len(dset))
    if n_tests != 0:
        dset_idx = sorted(random.sample(range(len(dset)), n_tests))

    dset = [dset[i] for i in dset_idx]

    x, y, _ = list(zip(*dset))
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    with torch.no_grad():
        net.reset()

        losses = []
        outs = []

        total_loss = torch.tensor(0.)
        for j in range(x.shape[1]):
            # run the step
            net_in = x[:,j].reshape(-1, net.args.L)
            net_out = net(net_in)
            outs.append(net_out)
            net_target = y[:,j].reshape(-1, net.args.Z)

            trial_losses = []
            for k in range(len(dset)):
                step_loss = criterion(net_out[k], net_target[k])
                trial_losses.append(step_loss)
            losses.append(np.array(trial_losses))

    losses = np.sum(losses, axis=0)
    z = torch.stack(outs, dim=1).squeeze()

    data = list(zip(dset_idx, x, y, z, losses))

    final_loss = np.mean(losses, axis=0)

    return data, final_loss

