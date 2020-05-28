import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse
import pdb
import sys
import pickle
import logging

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

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-E', '--n_epochs', type=int, default=10)

    parser.add_argument('--conv_type', type=str, choices=['patience', 'grad'], default='grad')
    parser.add_argument('--patience', type=int, default=1000, help='stop training if loss doesn\'t decrease')
    parser.add_argument('--grad_threshold', type=float, default=1e-4, help='stop training if grad is less than certain amount')

    parser.add_argument('--seed', type=int, help='seed for most of network')
    parser.add_argument('--reservoir_seed', type=int, help='seed for reservoir')

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

    dset = load_dataset(args.dataset)
    np.random.shuffle(dset)

    # val set is unused for now
    # cutoff = round(.9 * len(dset))
    # dset_train = dset[:cutoff]
    # dset_val = dset[cutoff:]

    dset_train = dset

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

        plot_checkpoint_path = os.path.join(log.log_dir, 'checkpoints.pkl')

    ix = 0
    losses = []
    max_abs_grads = []
    vis_samples = []
    running_min_error = float('inf')
    running_no_min = 0

    for e in range(args.n_epochs):
        np.random.shuffle(dset_train)
        for trial in dset_train:
            ix += 1
            # next data sample
            x = torch.from_numpy(trial[0]).float()
            y = torch.from_numpy(trial[1]).float()

            net.reset()
            optimizer.zero_grad()

            outs = []
            # the intermediate representation right before reservoir
            thals = []
            # the reservoir representations
            ress = []
            # individual losses for each timestep within a trial
            sublosses = []
            total_loss = torch.tensor(0.)
            ending = False
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
                    logging.info(f'iteration {ix}: is nan. ending')
                    ending = True
                    break
                sublosses.append(step_loss.item())
                total_loss += step_loss

            if ending:
                avg_loss = np.nan
                break

            total_loss.backward()

            # convergence based on maximum absolute value of gradient
            max_abs_grad = 0
            for p in train_params:
                p_max = torch.max(torch.abs(p.grad))
                if p_max > max_abs_grad:
                    max_abs_grad = p_max

            max_abs_grads.append(max_abs_grad)
            if args.conv_type == 'grad' and max_abs_grad < args.grad_threshold:
                logging.info(f'iteration {ix}: max absolute grad < {args.grad_threshold}. ending')
                break

            optimizer.step()
            losses.append(total_loss.item())

            # logging
            if ix % log_interval == 0:
                z = np.stack(outs).squeeze()
                # avg of the last 50 trials
                avg_loss = sum(losses[-log_interval:]) / log_interval
                avg_max_grad = sum(max_abs_grads[-log_interval:]) / log_interval
                logging.info(f'iteration {ix}\t| loss {avg_loss}\t| max abs grad {avg_max_grad}')

                # logging output
                if not args.no_log:
                    writer.writerow([ix, avg_loss])
                    vis_samples.append([ix, x.numpy(), y.numpy(), z, total_loss.item(), avg_loss])
                    # saving the model takes too much space so we just save one
                    model_path = os.path.join(log.checkpoint_dir, 'model.pth')
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    torch.save(net.state_dict(), model_path)

                    if os.path.exists(plot_checkpoint_path):
                        os.remove(plot_checkpoint_path)
                    with open(plot_checkpoint_path, 'wb') as f:
                        pickle.dump(vis_samples, f)

                # convergence based on no avg loss decrease after patience samples
                if args.conv_type == 'patience':
                    if avg_loss < running_min_error:
                        running_no_min = 0
                        running_min_error = avg_loss
                    else:
                        running_no_min += log_interval
                    if running_no_min > args.patience:
                        logging.info(f'iteration {ix}: no min for {args.patience} samples. ending')
                        break

        if running_no_min > args.patience or ending:
            break

    logging.info(f'END | iterations: {(ix // log_interval) * log_interval} | loss: {avg_loss}')

    with open(os.path.join(log.log_dir, 'checkpoints.pkl'), 'wb') as f:
        pickle.dump(vis_samples, f)

    csv_path.close()

    return avg_loss
    


if __name__ == '__main__':
    args = parse_args()

    if args.seed is None:
        args.seed = random.randint(1e9)
    if args.reservoir_seed is None:
        args.reservoir_seed = random.randint(1e9)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.slurm_id is not None:
        from parameters import apply_parameters
        args = apply_parameters(args.param_path, args)

    if not args.no_log:
        if args.slurm_id is not None:
            log = log_this(args, 'logs', os.path.join(args.name.split('_')[0], args.name.split('_')[1]), True)
        else:
            log = log_this(args, 'logs', args.name, True)

        logging.basicConfig(format='%(message)s', filename=log.run_log, level=logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(console)

    final_loss = train(args)

    if args.slurm_id is not None:
        # if running many jobs, then we gonna put the results into a csv
        csv_path = os.path.join('logs', args.name.split('_')[0] + '.csv')
        csv_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.writer(f, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if not csv_exists:
                writer.writerow(['slurm_id','N', 'D', 'O', 'epochs', 'lr', 'dset', 'loss'])
            writer.writerow([args.slurm_id, args.N, args.D, args.O, args.n_epochs, args.lr, args.dataset, final_loss])

    logging.shutdown()



