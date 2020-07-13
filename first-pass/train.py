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

from dataset import load_dataset
from reservoir import Network, Reservoir

from utils import log_this
from helpers import get_optimizer, get_criterion

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class Trainer:
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.net = Network(self.args)
        self.criterion = get_criterion(self.args)
        
        self.train_params = []
        for q in self.net.named_parameters():
            for p in self.args.train_parts:
                if q[0].find(p) != -1:
                    self.train_params.append(q[1])
        self.optimizer = get_optimizer(self.args, self.train_params)

        self.dset = load_dataset(self.args.dataset)
        
        self.log_interval = self.args.log_interval
        if not self.args.no_log:
            self.log = self.args.log
            self.vis_samples = []
            self.csv_path = open(os.path.join(self.log.run_dir, 'losses.csv'), 'a')
            self.writer = csv.writer(self.csv_path, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            self.writer.writerow(['ix', 'avg_loss'])
            self.plot_checkpoint_path = os.path.join(self.log.run_dir, 'checkpoints.pkl')
            self.model_path = os.path.join(log.run_dir, 'model.pth')

    def optimize_lbfgs(self, mode):
        if mode == 'pytorch':
            # don't use this. need to fix first if used
            best_loss = float('inf')
            best_loss_params = None
            for i in range(50):
                # just optimized based on a random 500 samples
                # don't do this!
                np.random.shuffle(self.dset)
                dset = np.asarray([x[:-1] for x in self.dset[:500]])
                x = torch.from_numpy(dset[:,0,:]).float()
                y = torch.from_numpy(dset[:,1,:]).float()

                def closure():
                    self.net.reset()
                    self.optimizer.zero_grad()
                    total_loss = torch.tensor(0.)
                    for j in range(x.shape[1]):
                        net_in = x[:,j]
                        net_out, val_res, val_thal = self.net(net_in)
                        net_target = y[:,j].reshape(-1, 1)

                        step_loss = self.criterion(net_out, net_target)
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
            W_f_total = self.args.L * self.args.D if 'W_f' in self.args.train_parts else 0
            W_ro_total = self.args.N * self.args.Z if 'W_ro' in self.args.train_parts else 0
            res_total = self.args.N * self.args.N if 'reservoir' in self.args.train_parts else 0
            W_u_total = self.args.D * self.args.N if 'reservoir' in self.args.train_parts else 0
            dset = np.asarray([x[:-1] for x in self.dset])
            x = torch.from_numpy(dset[:,0,:]).float()
            y = torch.from_numpy(dset[:,1,:]).float()

            # so that the callback for scipy.optimize.minimize knows what step it is on
            self.scipy_ix = 0
            vis_samples = []

            # turn a list into W_f (D x O) and W_ro (1 x N)
            # need this because LBFGS only takes in a list of params
            def vec_to_param(v):
                assert len(v) == W_f_total + W_ro_total + res_total + W_u_total
                W_f, W_ro, J, W_u = [None] * 4
                if 'W_f' in self.args.train_parts:
                    W_f = v[:W_f_total].reshape(self.args.D, self.args.L)
                if 'W_ro' in self.args.train_parts:
                    W_ro = v[W_f_total:W_f_total+W_ro_total].reshape(self.args.Z, self.args.N)
                if 'reservoir' in self.args.train_parts:
                    J = v[W_f_total+W_ro_total:W_f_total+W_ro_total+res_total].reshape(self.args.N, self.args.N)
                if 'reservoir' in self.args.train_parts:
                    W_u = v[W_f_total+W_ro_total+res_total:].reshape(self.args.N, self.args.D)
                return W_f, W_ro, J, W_u

            # this is what happens every iteration
            # run through all examples (x, y) and get loss, gradient
            def closure(v):
                W_f, W_ro, J, W_u = vec_to_param(v)
                if W_f is not None:
                    self.net.W_f.weight = nn.Parameter(torch.from_numpy(W_f).float())
                if W_ro is not None:
                    self.net.W_ro.weight = nn.Parameter(torch.from_numpy(W_ro).float())
                if J is not None:
                    self.net.reservoir.J.weight = nn.Parameter(torch.from_numpy(J).float())
                if W_u is not None:
                    self.net.reservoir.W_u.weight = nn.Parameter(torch.from_numpy(W_u).float())

                # need to do this so that burn in works
                # res state starting from same random seed for each iteration
                self.net.reset(res_state_seed=0)
                self.net.zero_grad()
                total_loss = torch.tensor(0.)
                for j in range(x.shape[1]):
                    net_in = x[:,j].reshape(-1, self.args.L)
                    net_out, val_res, val_thal = self.net(net_in)
                    net_target = y[:,j].reshape(-1, self.args.Z)

                    step_loss = self.criterion(net_out, net_target)
                    total_loss += step_loss

                total_loss.backward()

                grad_list = []
                if 'W_f' in self.args.train_parts:
                    grad_list.append(self.net.W_f.weight.grad.detach().numpy().reshape(-1))
                if 'W_ro' in self.args.train_parts:
                    grad_list.append(self.net.W_ro.weight.grad.detach().numpy().reshape(-1))
                if 'reservoir' in self.args.train_parts:
                    grad_list.append(self.net.reservoir.J.weight.grad.detach().numpy().reshape(-1))
                    grad_list.append(self.net.reservoir.W_u.weight.grad.detach().numpy().reshape(-1))
                    
                vec = np.concatenate(grad_list)
                post = np.float64(vec)

                return total_loss.item(), post

            # callback just does logging
            def callback(xk):
                if self.args.no_log:
                    return
                self.scipy_ix += 1
                if self.scipy_ix % self.log_interval == 0:
                    W_f, W_ro = vec_to_param(xk)
                    sample_n = random.randrange(len(dset))

                    self.net.reset(res_state_seed=0)
                    self.net.zero_grad()
                    outs = []
                    total_loss = torch.tensor(0.)
                    xs = x[sample_n,:]
                    ys = y[sample_n,:]
                    for j in range(xs.shape[0]):
                        net_in = xs[j].reshape(-1, self.args.L)
                        net_out, val_res, val_thal = self.net(net_in)
                        outs.append(net_out.item())
                        net_target = ys[j].reshape(-1, self.args.Z)
                        step_loss = self.criterion(net_out, net_target)
                        total_loss += step_loss

                    z = np.stack(outs).squeeze()
                    self.log_checkpoint(self.scipy_ix, xs.numpy(), ys.numpy(), z, total_loss.item(), total_loss.item())
                    logging.info(f'iteration {self.scipy_ix}\t| loss {total_loss.item():.3f}')

            # random initialization of input weights
            init_Wf = np.random.randn(self.args.D, self.args.L) / np.sqrt(self.args.L)
            init_Wro = np.random.randn(self.args.Z, self.args.N) / np.sqrt(self.args.N)
            init = np.concatenate((init_Wf.reshape(-1), init_Wro.reshape(-1)))
            optim_options = {
                'iprint': self.log_interval,
                'maxiter': self.args.maxiter,
                'ftol': 1e-12
            }
            optim = minimize(closure, init, method='L-BFGS-B', jac=True, callback=callback, options=optim_options)

            W_f_final, W_ro_final = vec_to_param(optim.x)
            error_final = optim.fun

            if not self.args.no_log:
                with open(os.path.join(self.log.run_dir, 'checkpoints.pkl'), 'wb') as f:
                    pickle.dump(self.vis_samples, f)
                self.csv_path.close()

            return error_final


    def log_checkpoint(self, ix, x, y, z, total_loss, avg_loss):
        self.writer.writerow([ix, avg_loss])
        self.csv_path.flush()
        # saving all checkpoints takes too much space so we just save one
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        torch.save(self.net.state_dict(), self.model_path)

        if self.args.loss == 'bce':
            z = sigmoid(z)

        self.vis_samples.append([ix, x, y, z, total_loss, avg_loss])
        if os.path.exists(self.plot_checkpoint_path):
            os.remove(self.plot_checkpoint_path)
        with open(self.plot_checkpoint_path, 'wb') as f:
            pickle.dump(self.vis_samples, f)

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
            net_out, val_res, val_thal = self.net(net_in)
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

        return total_loss, (x, y, outs, thals, ress, sublosses)

    def train(self):
        ix = 0

        # for convergence testing
        max_abs_grads = []
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

                loss, etc = self.iteration(x, y)
                self.optimizer.step()

                if loss == -1:
                    logging.info(f'iteration {ix}: is nan. ending')
                    ending = True
                    break

                running_loss += loss.item()
                mag = max([torch.max(torch.abs(p.grad)) for p in self.train_params])
                running_mag += mag             

                if ix % self.log_interval == 0:
                    outs = etc[2]
                    z = np.stack(outs).squeeze()
                    # avg of the last 50 trials
                    avg_loss = running_loss / self.log_interval
                    avg_max_grad = running_mag / self.log_interval
                    logging.info(f'iteration {ix}\t| loss {avg_loss:.3f}\t| max abs grad {avg_max_grad:.3f}')

                    if not self.args.no_log:
                        self.log_checkpoint(ix, etc[0].numpy(), etc[1].numpy(), z, running_loss, avg_loss)
                    running_loss = 0.0
                    running_mag = 0.0

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
                pickle.dump(self.vis_samples, f)

            self.csv_path.close()

        return avg_loss

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-L', type=int, default=1, help='')
    parser.add_argument('-D', type=int, default=5, help='')
    parser.add_argument('-N', type=int, default=20, help='')
    parser.add_argument('-Z', type=int, default=1, help='')

    parser.add_argument('--train_parts', type=str, nargs='+', choices=['reservoir', 'W_ro', 'W_f'], default=['W_ro', 'W_f'])
    parser.add_argument('--Wro_path', type=str, default=None, help='saved Wf')
    parser.add_argument('--Wf_path', type=str, default=None, help='saved Wf')
    parser.add_argument('--reservoir_path', type=str, default=None, help='saved reservoir. should be saved with seed tho')
    
    parser.add_argument('--res_init_type', type=str, default='gaussian', help='')
    parser.add_argument('--res_init_gaussian_std', type=float, default=1.5)
    parser.add_argument('--network_delay', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='datasets/rsg.pkl')

    parser.add_argument('--optimizer', choices=['adam', 'lbfgs-scipy', 'lbfgs-pytorch'], default='lbfgs-scipy')
    parser.add_argument('--loss', type=str, default='mse')

    # lbfgs-scipy arguments
    parser.add_argument('--maxiter', type=int, default=5000, help='limit to # iterations. lbfgs-scipy only')

    # adam arguments
    parser.add_argument('--conv_type', type=str, choices=['patience', 'grad'], default='grad', help='how to determine convergence. adam only')
    parser.add_argument('--patience', type=int, default=1000, help='stop training if loss doesn\'t decrease. adam only')
    parser.add_argument('--grad_threshold', type=float, default=1e-4, help='stop training if grad is less than certain amount. adam only')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. adam only')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train for. adam only')

    parser.add_argument('--seed', type=int, help='seed for most of network')
    parser.add_argument('--reservoir_seed', type=int, help='seed for reservoir')
    parser.add_argument('--reservoir_x_seed', type=int, default=0, help='seed for reservoir init hidden states. -1 for zero init')
    parser.add_argument('--reservoir_burn_steps', type=int, default=200, help='number of steps for reservoir to burn in')

    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--log_interval', type=int, default=50)

    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--param_path', type=str, default=None)
    parser.add_argument('--slurm_id', type=int, default=None)

    args = parser.parse_args()
    args.res_init_params = {}
    if args.res_init_type == 'gaussian':
        args.res_init_params['std'] = args.res_init_gaussian_std
    return args

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
    elif args.optimizer == 'adam':
        final_loss = trainer.train()

    if args.slurm_id is not None:
        # if running many jobs, then we gonna put the results into a csv
        csv_path = os.path.join('logs', args.name.split('_')[0] + '.csv')
        csv_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if args.optimizer == 'adam':
                if not csv_exists:
                    writer.writerow(['slurm_id','L', 'D', 'N', 'seed', 'rseed', 'xseed', 'epochs', 'lr', 'dset', 'loss'])
                writer.writerow([args.slurm_id, args.L, args.D, args.N, args.seed, args.reservoir_seed, args.reservoir_x_seed, args.n_epochs, args.lr, args.dataset, final_loss])
            elif args.optimizer == 'lbfgs-scipy':
                if not csv_exists:
                    writer.writerow(['slurm_id','L', 'D', 'N', 'seed', 'rseed', 'xseed', 'dset', 'loss'])
                writer.writerow([args.slurm_id, args.L, args.D, args.N, args.seed, args.reservoir_seed, args.reservoir_x_seed, args.dataset, final_loss])

    logging.shutdown()



