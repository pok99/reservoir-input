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

from network import BasicNetwork, Reservoir

from utils import log_this, load_rb, get_config, fill_undefined_args
from helpers import get_optimizer, get_scheduler, get_criteria, get_x_y_info

class Trainer:
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.net = BasicNetwork(self.args)

        # getting number of elements of every parameter
        self.n_params = {}
        self.train_params = []
        self.not_train_params = []
        logging.info('Training the following parameters:')
        for k,v in self.net.named_parameters():
            # k is name, v is weight
            found = False
            # filtering just for the parts that will be trained
            for part in self.args.train_parts:
                if part in k:
                    logging.info(f'  {k}')
                    self.n_params[k] = (v.shape, v.numel())
                    self.train_params.append(v)
                    found = True
                    break
            if not found:
                self.not_train_params.append(k)
        logging.info('Not training:')
        for k in self.not_train_params:
            logging.info(f'  {k}')

        self.criteria = get_criteria(self.args)
        self.optimizer = get_optimizer(self.args, self.train_params)
        self.scheduler = get_scheduler(self.args, self.optimizer)
        if self.args.dataset is None:
            # means we're using contexts
            dsets = []
            for d in self.args.contexts:
                dsets.append(load_rb(d))
        else:
            self.dset = load_rb(self.args.dataset)            

        # if using separate training and test sets, separate them out
        if self.args.same_test:
            self.train_set = self.dset
            self.test_set = self.dset
        else:
            np.random.shuffle(self.dset)
            cutoff = round(.9 * len(self.dset))
            self.train_set = self.dset[:cutoff]
            self.test_set = self.dset[cutoff:]
            logging.info(f'Using separate training ({cutoff}) and test ({len(self.dset) - cutoff}) sets.')
        
        self.log_interval = self.args.log_interval
        if not self.args.no_log:
            self.log = self.args.log
            self.run_id = self.args.log.run_id
            self.vis_samples = []
            self.csv_path = open(os.path.join(self.log.run_dir, f'losses_{self.run_id}.csv'), 'a')
            self.writer = csv.writer(self.csv_path, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            self.writer.writerow(['ix', 'avg_loss'])
            self.plot_checkpoint_path = os.path.join(self.log.run_dir, f'checkpoints_{self.run_id}.pkl')
            self.save_model_path = os.path.join(self.log.run_dir, f'model_{self.run_id}.pth')

    def optimize_lbfgs(self):
        x, y, info = get_x_y_info(args, self.dset)
        # so that the callback for scipy.optimize.minimize knows what step it is on
        self.scipy_ix = 0
        vis_samples = []

        # this is what happens every iteration
        # run through all examples (x, y) and get loss, gradient
        def closure(v):

            # setting the parameters in the network with the new values in v
            ind = 0
            for k,nums in self.n_params.items():
                # nums[0] is shape, nums[1] is number of elements
                weight = v[ind:ind+nums[1]].reshape(nums[0])
                self.net.state_dict()[k][:] = torch.Tensor(weight)
                ind += nums[1]

            # res state starting from same random seed for each iteration
            self.net.reset()
            self.net.zero_grad()

            # total_loss = torch.tensor(0.)
            total_loss, etc = self.run_trial(x, y, info, extras=True)
            total_loss.backward()

            # turn param grads into list
            grad_list = []
            for v in self.train_params:
                grad = v.grad.numpy().reshape(-1)
                grad_list.append(grad)
            vec = np.concatenate(grad_list)
            post = np.float64(vec)

            return total_loss.item() / len(x), post

        # callback just does logging
        def callback(xk):
            if self.args.no_log:
                return
            self.scipy_ix += 1
            if self.scipy_ix % self.log_interval == 0:
                sample_n = random.randrange(len(self.dset))

                with torch.no_grad():
                    self.net.reset()
                    self.net.zero_grad()
                    outs = []
                    total_loss = torch.tensor(0.)

                    xs = x[sample_n,:]
                    ys = y[sample_n,:]
                    pdb.set_trace()
                    for j in range(xs.shape[0]):
                        net_out, step_loss, _ = self.run_iteration(xs[j], ys[j])
                        outs.append(net_out.item())
                        total_loss += step_loss

                    z = np.stack(outs).squeeze()
                    self.log_checkpoint(self.scipy_ix, xs.numpy(), ys.numpy(), z, total_loss.item(), total_loss.item())

                    logging.info(f'iteration {self.scipy_ix}\t| loss {total_loss.item():.3f}')

        # getting the initial values to put into the algorithm
        init_list = []
        for v in self.train_params:
            init_list.append(v.detach().numpy().reshape(-1))
        init = np.concatenate(init_list)

        optim_options = {
            'iprint': self.log_interval,
            'maxiter': self.args.maxiter,
            'ftol': 1e-16
        }
        optim = minimize(closure, init, method='L-BFGS-B', jac=True, callback=callback, options=optim_options)

        error_final = optim.fun
        n_iters = optim.nit

        if not self.args.no_log:
            self.log_model(name='model_final.pth')
            if self.args.log_checkpoint_samples:
                with open(self.plot_checkpoint_path, 'wb') as f:
                    pickle.dump(self.vis_samples, f)
            self.csv_path.close()

        return error_final, n_iters

    def log_model(self, ix=0, name=None):
        # if we want to save a particular name, just do it and leave
        if name is not None:
            model_path = os.path.join(self.log.run_dir, name)
            if os.path.exists(model_path):
                os.remove(model_path)
            torch.save(self.net.state_dict(), model_path)
            return
        # saving all checkpoints takes too much space so we just save one model at a time, unless we explicitly specify it
        if self.args.log_checkpoint_models:
            self.save_model_path = os.path.join(self.log.checkpoint_dir, f'model_{ix}.pth')
        elif os.path.exists(self.save_model_path):
            os.remove(self.save_model_path)
        torch.save(self.net.state_dict(), self.save_model_path)

    def log_checkpoint(self, ix, x, y, z, total_loss, avg_loss):
        self.writer.writerow([ix, avg_loss])
        self.csv_path.flush()

        self.log_model(ix)

        # we can save individual samples at each checkpoint, that's not too bad space-wise
        if self.args.log_checkpoint_samples:
            self.vis_samples.append([ix, x, y, z, total_loss, avg_loss])
            if os.path.exists(self.plot_checkpoint_path):
                os.remove(self.plot_checkpoint_path)
            with open(self.plot_checkpoint_path, 'wb') as f:
                pickle.dump(self.vis_samples, f)


    # runs an iteration where we want to match a certain trajectory
    def run_trial(self, x, y, info, extras=False):
        total_loss = 0.
        outs = []
        for j in range(x.shape[1]):
            net_in = x[:,j].reshape(-1, self.args.L)
            net_out, extras = self.net(net_in, extras=True)
            outs.append(net_out)

        net_outs = torch.cat(outs, dim=1)
        net_targets = y
        for c in self.criteria:
            total_loss += c(net_outs, net_targets, info)

        if extras:
            etc = {'outs': net_outs,}
            return total_loss, etc
        return total_loss

    def train_iteration(self, x, y, info):
        self.net.reset(self.args.res_x_init)
        self.optimizer.zero_grad()

        total_loss, etc = self.run_trial(x, y, info, extras=True)
        total_loss.backward()

        etc = {
            'ins': x,
            'goals': y,
            'outs': etc['outs'].detach()
        }
        return total_loss.item(), etc

    def test(self, n=0):
        if n != 0:
            assert n <= len(self.test_set)
            batch_idxs = np.random.choice(len(self.test_set), n)
            batch = [self.test_set[i] for i in batch_idxs]
        else:
            batch = self.test_set

        with torch.no_grad():
            x, y, info = get_x_y_info(args, batch)
            self.net.reset(self.args.res_x_init)
            total_loss, etc = self.run_trial(x, y, info, extras=True)

        return total_loss.item() / len(batch), etc

    def train(self, ix_callback=None):
        ix = 0

        its_p_epoch = len(self.train_set) // self.args.batch_size
        logging.info(f'Training set size {len(self.train_set)} | batch size {self.args.batch_size} --> {its_p_epoch} iterations / epoch')

        # for convergence testing
        running_min_error = float('inf')
        running_no_min = 0

        running_loss = 0.0
        ending = False
        for e in range(self.args.n_epochs):
            np.random.shuffle(self.train_set)
            epoch_idx = 0
            while epoch_idx < its_p_epoch:
                epoch_idx += 1
                batch = self.train_set[(epoch_idx-1) * self.args.batch_size:epoch_idx * self.args.batch_size]
                if len(batch) < self.args.batch_size:
                    break
                ix += 1

                x, y, info = get_x_y_info(args, batch)
                iter_loss, etc = self.train_iteration(x, y, info)

                if ix_callback is not None:
                    ix_callback(iter_loss, etc)
                self.optimizer.step()

                if iter_loss == -1:
                    logging.info(f'iteration {ix}: is nan. ending')
                    ending = True
                    break

                running_loss += iter_loss

                if ix % self.log_interval == 0:
                    z = etc['outs'].numpy().squeeze()
                    avg_loss = running_loss / self.args.batch_size / self.log_interval
                    test_loss, test_etc = self.test()
                    log_arr = [
                        f'iteration {ix}',
                        f'loss {avg_loss:.3f}',
                        f'test loss {test_loss:.3f}',
                    ]
                    log_str = '\t| '.join(log_arr)
                    logging.info(log_str)

                    if not self.args.no_log:
                        self.log_checkpoint(ix, etc['ins'].numpy(), etc['goals'].numpy(), z, running_loss, avg_loss)
                    running_loss = 0.0

                    # convergence based on no avg loss decrease after patience samples
                    if test_loss < running_min_error:
                        running_no_min = 0
                        running_min_error = test_loss
                        if not self.args.no_log:
                            self.log_model(name='model_best.pth')
                    else:
                        running_no_min += self.log_interval
                    if running_no_min > self.args.patience:
                        logging.info(f'iteration {ix}: no min for {args.patience} samples. ending')
                        ending = True
                if ending:
                    break
            logging.info(f'Finished dataset epoch {e+1}')
            if self.scheduler is not None:
                self.scheduler.step()
            if ending:
                break

        if not self.args.no_log and self.args.log_checkpoint_samples:
            # for later visualization of outputs over timesteps
            with open(self.plot_checkpoint_path, 'wb') as f:
                pickle.dump(self.vis_samples, f)

            self.csv_path.close()

        logging.info(f'END | iterations: {(ix // self.log_interval) * self.log_interval} | best loss: {running_min_error}')
        return running_min_error, ix

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-L', type=int, default=2, help='latent input dimension')
    parser.add_argument('-T', type=int, default=1, help='task dimension')
    parser.add_argument('-D', type=int, default=5, help='intermediate dimension')
    parser.add_argument('-N', type=int, default=50, help='number of neurons in reservoir')
    parser.add_argument('-Z', type=int, default=1, help='output dimension')

    parser.add_argument('--net', type=str, default='basic', choices=['basic'])

    parser.add_argument('--train_parts', type=str, nargs='+', default=['W_ro', 'W_f'])
    # parser.add_argument('--stride', type=int, default=1, help='stride of the W_f')
    
    # make sure model_config path is specified if you use any paths! it ensures correct dimensions, bias, etc.
    parser.add_argument('--model_config_path', type=str, default=None, help='config path corresponding to model load path')
    parser.add_argument('--model_path', type=str, default=None, help='start training from certain model. superseded by below')
    parser.add_argument('--Wro_path', type=str, default=None, help='start training from certain Wro')
    parser.add_argument('--Wf_path', type=str, default=None, help='start training from certain Wf')
    parser.add_argument('--J_path', type=str, default=None, help='saved reservoir. should be saved with seed tho')
    
    # network manipulation
    parser.add_argument('--res_init_type', type=str, default='gaussian', help='')
    parser.add_argument('--res_init_gaussian_std', type=float, default=1.5)
    parser.add_argument('--res_noise', type=float, default=0)
    parser.add_argument('--x_noise', type=float, default=0)
    parser.add_argument('--m_noise', type=float, default=0)
    parser.add_argument('--no_bias', action='store_true')
    parser.add_argument('--out_act', type=str, default=None, help='output activation')

    parser.add_argument('-d', '--dataset', type=str, default='datasets/rsg-sohn.pkl')
    # need same number of inputs for the following two arguments. using them overwrites -d dataset
    parser.add_argument('-c', '--contexts', type=str, nargs='+', default=[], help='list of datasets to use for context training')
    parser.add_argument('-n', '--context_names', type=str, nargs='+', default=[], help='names of datasets used for context training')
    # high-level arguments that control dataset manipulations
    parser.add_argument('--same_signal', action='store_true', help='use 1d input instead of separate ready/set pulses')
    parser.add_argument('--same_test', action='store_true', help='use entire dataset for both training and testing')
    
    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'rmsprop', 'lbfgs'], default='adam')

    # adam parameters
    parser.add_argument('--batch_size', type=int, default=1, help='size of minibatch used')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. adam only')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train for. adam only')
    parser.add_argument('--conv_type', type=str, choices=['patience', 'grad'], default='patience', help='how to determine convergence. adam only')
    parser.add_argument('--patience', type=int, default=2000, help='stop training if loss doesn\'t decrease. adam only')
    parser.add_argument('--l2_reg', type=float, default=0, help='amount of l2 regularization')
    parser.add_argument('--s_rate', default=None, type=float, help='scheduler rate. dont use for no scheduler')
    parser.add_argument('--losses', type=str, nargs='+', choices=['mse', 'bce', 'mse-w', 'bce-w', 'mse-g', 'mse-w2'], default=[])

    # adam lambdas
    parser.add_argument('--l1', type=float, default=1, help='weight of normal loss')
    parser.add_argument('--l2', type=float, default=0.5, help='weight of secondary (windowed/goal) loss')
    parser.add_argument('--l3', type=float, default=100, help='bce: weight of positive examples')
    parser.add_argument('--l4', type=float, default=10, help='bce-w: weight of positive examples')

    # lbfgs-scipy parameters
    parser.add_argument('--maxiter', type=int, default=10000, help='limit to # iterations. lbfgs-scipy only')

    # seeds
    parser.add_argument('--seed', type=int, help='seed for most of network')
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
    parser.add_argument('--param_path', type=str, default=None)
    parser.add_argument('--slurm_id', type=int, default=None)

    args = parser.parse_args()
    args.res_init_params = {}
    if args.res_init_type == 'gaussian':
        args.res_init_params['std'] = args.res_init_gaussian_std
    args.bias = not args.no_bias
    return args

def adjust_args(args):
    # don't use logging.info before we initialize the logger!! or else stuff is gonna fail

    # setting seeds
    if args.res_seed is None:
        args.res_seed = random.randrange(1e6)
    if args.seed is None:
        args.seed = random.randrange(1e6)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # dealing with slurm. do this first!! before anything else other than seed setting, which we want to override
    if args.slurm_id is not None:
        from parameters import apply_parameters
        args = apply_parameters(args.param_path, args)

    # in case we are loading from a model
    # if we don't use this we might end up with an error when loading model
    # in case we are loading from a model
    # if we don't use this we might end up with an error when loading model
    if args.model_path is not None:
        config = get_config(args.model_path)
        args = fill_undefined_args(args, config, overwrite_none=True)
        enforce_same = ['N', 'D', 'L', 'Z', 'T', 'net', 'bias', 'use_reservoir']
        for v in enforce_same:
            if v in config and args.__dict__[v] != config[v]:
                print(f'Warning: based on config, changed {v} from {args.__dict__[v]} -> {config[v]}')
                args.__dict__[v] = config[v]

    # shortcut for specifying train everything including reservoir
    if args.train_parts == ['all']:
        args.train_parts = ['']

    # set the dataset
    config = get_config(args.dataset, ctype='data', to_bunch=True)
    if config.t_type == 'rsg-bin':
        args.dset_type = 'rsg-bin'
        args.out_act = 'none'
    elif config.t_type == 'rsg-sohn':
        args.dset_type = 'rsg-sohn'
        args.out_act = 'exp'
    elif config.t_type == 'copy':
        args.dset_type = 'copy'
    else:
        args.dset_type = 'unknown'

    if args.same_signal:
        args.L = 1

    args.n_contexts = 0
    assert len(args.contexts) == len(args.context_names) or len(args.context_names) == 0
    if len(args.contexts) > 0:
        args.n_contexts = len(args.contexts)
        args.dataset = 'None'

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
    logging.info(f'Initialized trainer. Using optimizer {args.optimizer}.')
    n_iters = 0
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
            labels_csv = ['slurm_id', 'N', 'D', 'seed', 'rseed', 'mnoise', 'rnoise', 'dset', 'niter', 'tparts', 'loss']
            vals_csv = [
                args.slurm_id, args.N, args.D, args.seed,
                args.res_seed, args.m_noise, args.res_noise,
                args.dataset, n_iters, '-'.join(args.train_parts), best_loss
            ]
            if args.optimizer == 'adam':
                labels_csv.extend(['lr', 'epochs'])
                vals_csv.extend([args.lr, args.n_epochs])

            if not csv_exists:
                writer.writerow(labels_csv)
            writer.writerow(vals_csv)

    logging.shutdown()



