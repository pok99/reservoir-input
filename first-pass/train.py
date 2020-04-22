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


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-N', default=10, help='')
    parser.add_argument('-D', default=2, help='')
    parser.add_argument('--res_init_type', default='gaussian', help='')
    parser.add_argument('--res_init_gaussian_std', default=1.5)

    # todo: arguments for res init parameters

    parser.add_argument('-O', default=1, help='')


    args = parser.parse_args()
    args.res_init_params = {}
    if args.res_init_type == 'gaussian':
        args.res_init_params['std'] = args.res_init_gaussian_std
    return args



def main():
    args = parse_args()

    dset = load_dataset('data/rsg_1.pkl')

    net = Network(args)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.1)

    for p in net.parameters():
        print(p)

    for q in net.named_parameters():
        print(q)
    sys.exit()

    ix = 0
    for trial in dset:
        ix += 1
        # next data sample
        x, y = torch.from_numpy(trial[0]).float(), torch.from_numpy(trial[1]).long()

        net.reset()
        optimizer.zero_grad()

        outs = []
        total_loss = torch.tensor(0.)
        for j in range(x.shape[0]):
            net_in = x[j].unsqueeze(-1)
            net_out = net(net_in)
            net_target = y[j].unsqueeze(0)
            outs.append(net_out)
            step_loss = criterion(net_out.transpose(0,1), net_target)
            total_loss += step_loss

        total_loss.backward()
        optimizer.step()
        print(total_loss)


        if ix > 60:
            break

    print(torch.stack(outs).squeeze().max(1)[1].flatten())

    pdb.set_trace()


if __name__ == '__main__':
    main()