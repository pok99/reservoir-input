import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse
import pdb
import sys
import pickle

from dataset import load_dataset
from reservoir import Network, Reservoir

from utils import *

log_interval = 50


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-N', type=int, default=20, help='')
    parser.add_argument('-D', type=int, default=5, help='')
    parser.add_argument('--res_init_type', default='gaussian', help='')
    parser.add_argument('--res_init_gaussian_std', default=1.5)
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--dataset', default='data/rsg_tl300.pkl')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-E', '--n_epochs', type=int, default=1)

    # todo: arguments for res init parameters

    parser.add_argument('-O', default=1, help='')

    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--param_path', type=str, default=None)
    parser.add_argument('--slurm_id', type=int, default=None)


    args = parser.parse_args()
    args.res_init_params = {}
    if args.res_init_type == 'gaussian':
        args.res_init_params['std'] = args.res_init_gaussian_std
    return args


def train(args):
    

    if not args.no_log:
        log = log_this(args, 'logs', args.name, False)

    dset = load_dataset(args.dataset)

    net = Network(args)

    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 10],dtype=torch.float), reduction='sum')
    criterion = nn.MSELoss()
    
    # don't train the reservoir
    # at the moment, trains W_f and W_o
    train_params = []
    for q in net.named_parameters():
        if q[0].split('.')[0] != 'reservoir':
            train_params.append(q[1])

    optimizer = optim.Adam(train_params, lr=args.lr)
    #optimizer = optim.SGD(train_params, lr=args.lr)

    # set up logging
    if not args.no_log:
        csv_path = open(os.path.join(log.log_dir, 'losses.csv'), 'a')
        writer = csv.writer(csv_path, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ix','avg_loss'])

    ix = 0
    losses = []
    vis_samples = []
    for e in range(args.n_epochs):
        np.random.shuffle(dset)
        for trial in dset:
            ix += 1
            # next data sample
            x = torch.from_numpy(trial[0]).float()
            y = torch.from_numpy(trial[1]).float()

            net.reset()
            optimizer.zero_grad()

            outs = []
            # the intermediate representation right before reservoir
            thals = []
            ress = []
            sublosses = []
            total_loss = torch.tensor(0.)
            ending = False
            # for all the steps in this trial
            for j in range(x.shape[0]):
                # run the step
                net_in = x[j].unsqueeze(0)
                net_out, val_thal, val_res = net(net_in)

                # this is the desired output
                net_target = y[j].unsqueeze(0)
                outs.append(net_out.item())
                thals.append(list(val_thal.detach().numpy().squeeze()))
                ress.append(list(val_res.detach().numpy().squeeze()))

                # this is the loss from the step
                step_loss = criterion(net_out.view(1), net_target)
                if np.isnan(step_loss.item()):
                    print('is nan. ending')
                    ending = True
                sublosses.append(step_loss.item())
                total_loss += step_loss

            if ending:
                break

            total_loss.backward()
            optimizer.step()

            losses.append(total_loss.item())

            if ix % log_interval == 0:
                z = np.stack(outs).squeeze()
                # avg of the last 50 trials
                avg_loss = sum(losses[-log_interval:]) / log_interval
                print(f'iteration {ix}; loss ', avg_loss)

                # logging output
                if not args.no_log:
                    writer.writerow([ix, avg_loss])
                    vis_samples.append([ix, x.numpy(), y.numpy(), z, total_loss.item(), avg_loss])                


    with open(os.path.join(log.log_dir, 'checkpoints.pkl'), 'wb') as f:
        pickle.dump(vis_samples, f)

    csv_path.close()
    



if __name__ == '__main__':
    args = parse_args()

    if args.slurm_id is not None:
        from parameters import apply_parameters
        args = apply_parameters(param_path, args)

    train(args)

