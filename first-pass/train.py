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

from dataset import load_dataset
from reservoir import Network, Reservoir

from utils import *
from helpers import get_optimizer

log_interval = 50

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-N', type=int, default=20, help='')
    parser.add_argument('-D', type=int, default=5, help='')
    parser.add_argument('--res_init_type', default='gaussian', help='')
    parser.add_argument('--res_init_gaussian_std', default=1.5)
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--dataset', default='data/rsg_tl100_sc1.pkl')

    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-E', '--n_epochs', type=int, default=10)

    parser.add_argument('--conv_type', type=str, choices=['patience', 'grad'], default='grad')
    parser.add_argument('--patience', type=int, default=1000, help='stop training if loss doesn\'t decrease')
    parser.add_argument('--grad_threshold', type=float, default=1e-4, help='stop training if grad is less than certain amount')

    parser.add_argument('--seed', type=int, help='seed for most of network')
    parser.add_argument('--reservoir_seed', type=int, help='seed for reservoir')

    parser.add_argument('-O', default=1, help='')

    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--param_path', type=str, default=None)
    parser.add_argument('--slurm_id', type=int, default=None)


    args = parser.parse_args()
    args.res_init_params = {}
    if args.res_init_type == 'gaussian':
        args.res_init_params['std'] = args.res_init_gaussian_std
    return args


class Trainer:
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.net = Network(self.args)
        self.criterion = nn.MSELoss()

        self.train_params = []
        for q in self.net.named_parameters():
            if q[0].split('.')[0] != 'reservoir':
                self.train_params.append(q[1])
        self.optimizer = get_optimizer(self.args, self.train_params)

        self.dset = load_dataset(self.args.dataset)
        
        self.log_interval = self.args.log_interval
        if not self.args.no_log:
            self.log = self.args.log
            self.csv_path = open(os.path.join(self.log.run_dir, 'losses.csv'), 'a')
            self.writer = csv.writer(self.csv_path, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            self.writer.writerow(['ix','avg_loss'])
            self.plot_checkpoint_path = os.path.join(self.log.run_dir, 'checkpoints.pkl')
            self.model_path = os.path.join(log.run_dir, 'model.pth')


    def iteration(self, x, y):
        self.net.reset()
        self.optimizer.zero_grad()

        outs = []
        # the intermediate representation right before reservoir
        thals = []
        # the reservoir representations
        ress = []
        # individual losses for each timestep within a trial
        sublosses = []
        total_loss = torch.tensor(0.)
        for j in range(x.shape[0]):
            # run the step
            net_in = x[j].unsqueeze(0)
            net_out, val_thal, val_res = self.net(net_in)
            net_target = y[j].unsqueeze(0)

            outs.append(net_out.item())
            thals.append(list(val_thal.detach().numpy().squeeze()))
            ress.append(list(val_res.detach().numpy().squeeze()))

            step_loss = self.criterion(net_out.view(1), net_target)
            if np.isnan(step_loss.item()):
                return -1, (outs, thals, ress, sublosses)
            sublosses.append(step_loss.item())
            total_loss += step_loss

        total_loss.backward()

        return total_loss, (outs, thals, ress, sublosses)

    def optimize_lbfgs(self, mode):
        if mode == 'pytorch':
            best_loss = float('inf')
            best_loss_params = None   
            for i in range(50):
                #dset = np.asarray([x[:-1] for x in self.dset[i * 100:(i+1) * 100]])
                np.random.shuffle(self.dset)
                dset = np.asarray([x[:-1] for x in self.dset[:500]])
                x = torch.from_numpy(dset[:,0,:]).float()
                y = torch.from_numpy(dset[:,1,:]).float()

                def closure():
                    self.net.reset()
                    self.optimizer.zero_grad()
                    total_loss = torch.tensor(0.)
                    for j in range(x.shape[1]):
                        # run the step
                        net_in = x[:,j]
                        net_out, val_thal, val_res = self.net(net_in)
                        net_target = y[:,j].reshape(-1, 1)

                        # outs.append(net_out.item())
                        # thals.append(list(val_thal.detach().numpy().squeeze()))
                        # ress.append(list(val_res.detach().numpy().squeeze()))

                        step_loss = self.criterion(net_out, net_target)
                        # if np.isnan(step_loss.item()):
                        #     return -1, (outs, thals, ress, sublosses)
                        # sublosses.append(step_loss.item())
                        total_loss += step_loss

                    total_loss.backward()
                    return total_loss

                self.optimizer.step(closure)
                loss = closure()
                
                if loss < best_loss:
                    print(i, loss.item(), 'new best loss')
                    best_loss = loss
                    best_loss_params = self.net.state_dict()
                else:
                    print(i, loss.item(), 'nope')
                    self.net.load_state_dict(best_loss_params)

        elif mode == 'scipy':
            W_f_total = self.args.O * self.args.D
            W_ro_total = self.args.N
            dset = np.asarray([x[:-1] for x in self.dset])
            x = torch.from_numpy(dset[:,0,:]).float()
            y = torch.from_numpy(dset[:,1,:]).float()
            def closure():
                self.net.reset()
                self.net.zero_grad()
                total_loss = torch.tensor(0.)
                for j in range(x.shape[1]):
                    # run the step
                    net_in = x[:,j]
                    net_out, val_thal, val_res = self.net(net_in)
                    net_target = y[:,j].reshape(-1, 1)

                    # outs.append(net_out.item())
                    # thals.append(list(val_thal.detach().numpy().squeeze()))
                    # ress.append(list(val_res.detach().numpy().squeeze()))

                    step_loss = self.criterion(net_out, net_target)
                    # if np.isnan(step_loss.item()):
                    #     return -1, (outs, thals, ress, sublosses)
                    # sublosses.append(step_loss.item())
                    total_loss += step_loss

                total_loss.backward()
                return total_loss

            def closure2(params):
                
                assert len(params) == W_f_total + W_ro_total
                W_f = params[:W_f_total].reshape(self.args.D, self.args.O)
                W_ro = params[W_f_total:].reshape(1, self.args.N)

                self.net.W_f.weight = nn.Parameter(torch.from_numpy(W_f).float())
                self.net.W_ro.weight = nn.Parameter(torch.from_numpy(W_ro).float())

                loss = closure()
                post_Wf = self.net.W_f.weight.grad.detach().numpy().reshape(-1)
                post_Wro = self.net.W_ro.weight.grad.detach().numpy().reshape(-1)
                post = np.concatenate((post_Wf, post_Wro))

                post = np.float64(post)
                
                return loss.item(), post

            init_Wf = np.random.randn(self.args.D, self.args.O) / np.sqrt(self.args.O)  # random initialization of input weights
            init_Wro = np.random.randn(1, self.args.N) / np.sqrt(self.args.N)
            init = np.concatenate((init_Wf.reshape(-1), init_Wro.reshape(-1)))
            optim = minimize(closure2, init, method='L-BFGS-B', jac=True, options={'iprint': 50})

            final = optim.x
            W_f_final = optim.x[:W_f_total].reshape(self.args.D, self.args.O)
            W_ro_final = optim.x[W_f_total:].reshape(1, self.args.N)
            error_final = optim.fun

            pdb.set_trace()


    def train(self):
        ix = 0

        # for convergence testing
        max_abs_grads = []
        vis_samples = []
        running_min_error = float('inf')
        running_no_min = 0

        running_loss = 0.0
        running_mag = 0.0
        avg_loss = -1
        ending = False
        for e in range(self.args.n_epochs):
            np.random.shuffle(self.dset)
            for trial in self.dset:
                ix += 1
                x = torch.from_numpy(trial[0]).float()
                y = torch.from_numpy(trial[1]).float()

                def closure():
                    loss, etc = self.iteration(x, y)
                    return loss

                if self.args.optimizer == 'adam':
                    loss, etc = self.iteration(x, y)
                    self.optimizer.step()
                elif self.args.optimizer == 'lbfgs':
                    self.optimizer.step(closure)
                    loss, etc = self.iteration(x, y)
                    init = np.random.randn(args.N, args.D) / np.sqrt(d)  # random initialization of input weights
                    #optim = minimize(loss, init.reshape(-1), method='L-BFGS-B', jac=True, options={'iprint': 10})  # the last argument just tells it to print stuff every 10 

                if loss == -1:
                    logging.info(f'iteration {ix}: is nan. ending')
                    ending = True
                    break

                running_loss += loss.item()
                mag = max([torch.max(torch.abs(p.grad)) for p in self.train_params])
                running_mag += mag             

                if ix % self.log_interval == 0:
                    outs = etc[0]
                    z = np.stack(outs).squeeze()
                    # avg of the last 50 trials
                    avg_loss = running_loss / self.log_interval
                    avg_max_grad = running_mag / self.log_interval
                    running_loss = 0.0
                    running_mag = 0.0
                    logging.info(f'iteration {ix}\t| loss {avg_loss:.3f}\t| max abs grad {avg_max_grad:.3f}')

                    if not self.args.no_log:
                        self.writer.writerow([ix, avg_loss])
                        # saving all checkpoints takes too much space so we just save one
                        if os.path.exists(self.model_path):
                            os.remove(self.model_path)
                        torch.save(self.net.state_dict(), self.model_path)

                        vis_samples.append([ix, x.numpy(), y.numpy(), z, total_loss.item(), avg_loss])
                        if os.path.exists(self.plot_checkpoint_path):
                            os.remove(self.plot_checkpoint_path)
                        with open(self.plot_checkpoint_path, 'wb') as f:
                            pickle.dump(vis_samples, f)

                    # convergence based on no avg loss decrease after patience samples
                    if self.args.conv_type == 'patience':
                        if avg_loss < running_min_error:
                            running_no_min = 0
                            running_min_error = avg_loss
                        else:
                            running_no_min += self.log_interval
                        if running_no_min > self.args.patience:
                            logging.info(f'iteration {ix}: no min for {args.patience} samples. ending')
                            ending = True
                    elif self.args.conv_type == 'grad':
                        if avg_max_grad < self.args.grad_threshold:
                            logging.info(f'iteration {ix}: max absolute grad < {args.grad_threshold}. ending')
                            ending = True

                if ending:
                    break
            if ending:
                break

        logging.info(f'END | iterations: {(ix // self.log_interval) * self.log_interval} | loss: {avg_loss}')

        if not self.args.no_log:
            # for later visualization of outputs over timesteps
            with open(os.path.join(self.log.run_dir, 'checkpoints.pkl'), 'wb') as f:
                pickle.dump(vis_samples, f)

            self.csv_path.close()

        return avg_loss


if __name__ == '__main__':
    args = parse_args()

    if args.seed is None:
        args.seed = random.randrange(1e6)
    if args.reservoir_seed is None:
        args.reservoir_seed = random.randrange(1e6)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.slurm_id is not None:
        from parameters import apply_parameters
        args = apply_parameters(args.param_path, args)

    if not args.no_log:
        if args.slurm_id is not None:
            log = log_this(args, 'logs', os.path.join(args.name.split('_')[0], args.name.split('_')[1]), checkpoints=False)
        else:
            log = log_this(args, 'logs', args.name, checkpoints=False)

        logging.basicConfig(format='%(message)s', filename=log.run_log, level=logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(console)
        args.log = log
    else:
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    trainer = Trainer(args)
    if args.optimizer == 'lbfgs-scipy':
        final_loss = trainer.optimize_lbfgs('scipy')
    elif args.optimizer == 'lbfgs-pytorch':
        final_loss = trainer.optimize_lbfgs('pytorch')
    else:
        final_loss = trainer.train()

    if args.slurm_id is not None:
        # if running many jobs, then we gonna put the results into a csv
        csv_path = os.path.join('logs', args.name.split('_')[0] + '.csv')
        csv_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.writer(f, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if not csv_exists:
                writer.writerow(['slurm_id','N', 'D', 'O', 'seed', 'rseed', 'epochs', 'lr', 'dset', 'loss'])
            writer.writerow([args.slurm_id, args.N, args.D, args.O, args.seed, args.reservoir_seed, args.n_epochs, args.lr, args.dataset, final_loss])

    logging.shutdown()



