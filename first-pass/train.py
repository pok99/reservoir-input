import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse
import pdb
import sys

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

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=1)

    # todo: arguments for res init parameters

    parser.add_argument('-O', default=1, help='')


    args = parser.parse_args()
    args.res_init_params = {}
    if args.res_init_type == 'gaussian':
        args.res_init_params['std'] = args.res_init_gaussian_std
    return args



def main():
    args = parse_args()

    dset = load_dataset('data/rsg_tl10.pkl')

    net = Network(args)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 10],dtype=torch.float), reduction='sum')
    
    # don't train the reservoir
    # at the moment, trains W_f and W_o
    train_params = []
    for q in net.named_parameters():
        if q[0].split('.')[0] != 'reservoir':
            train_params.append(q[1])

    optimizer = optim.Adam(train_params, lr=args.lr)

    ix = 0
    losses = []
    for e in range(args.n_epochs):
        np.random.shuffle(dset)
        for trial in dset:
            ix += 1
            # next data sample
            x, y = torch.from_numpy(trial[0]).float(), torch.from_numpy(trial[1]).long()

            net.reset()
            optimizer.zero_grad()

            outs = []
            thals = []
            sublosses = []
            total_loss = torch.tensor(0.)
            # for all the steps in this trial
            for j in range(x.shape[0]):
                # run the step
                net_in = x[j].unsqueeze(0)
                net_out, thal = net(net_in)
                # this is the desired output
                net_target = y[j].unsqueeze(0)
                outs.append(net_out)
                thals.append(list(thal.detach().numpy().squeeze()))
                # this is the loss from the step
                step_loss = criterion(net_out.transpose(0,1), net_target)
                sublosses.append(step_loss.item())
                total_loss += step_loss

            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

            if ix % log_interval == 0:
                # avg of the last 50 trials
                print(f'iteration {ix + len(dset) * e}; loss ', sum(losses[-log_interval:]) / log_interval)
                print('input:  ', (x.long() + y).numpy())
                print('output: ', torch.stack(outs).squeeze().max(1)[1].detach().numpy().flatten())
                # print('output: ', torch.stack(outs).squeeze().detach().numpy())
                # print('thals: ', thals)
                #print('outs ', [list(i.detach().numpy().squeeze()) for i in outs])
                #print(sublosses)
                #pdb.set_trace()

            # if ix % (log_interval * 5) == 0:
            #     pdb.set_trace()



if __name__ == '__main__':
    main()