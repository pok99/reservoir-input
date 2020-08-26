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

from network import BasicNetwork, StateNet, Reservoir

from utils import log_this, load_rb
from helpers import get_optimizer, get_criterion

class Trainer:
    def __init__(self, args):
        super().__init__()

        self.args = args

        if self.args.net == 'basic':
            self.net = BasicNetwork(self.args)
        elif self.args.net == 'state':
            self.net = StateNet(self.args)

        # load any specified model parameters into the network
        if args.model_path is not None:
            m_dict = torch.load(args.model_path)
            self.net.load_state_dict(m_dict)
            logging.info('Loaded model file.')
        if args.Wf_path is not None:
            m_dict = torch.load(args.Wf_path)
            self.net.W_f.weight = m_dict['W_f.weight']
            if 'W_f.bias' in m_dict:
                self.net.W_f.bias = m_dict['W_f.bias']
        if args.Wro_path is not None:
            m_dict = torch.load(args.Wro_path)
            self.net.W_ro.weight = m_dict['W_ro.weight']
            if 'W_ro.bias' in m_dict:
                self.net.W_ro.bias = m_dict['W_ro.bias']
        if args.reservoir_path is not None:
            m_dict = torch.load(args.Wro_path)
            self.net.reservoir.J.weight = m_dict['reservoir.J.weight']
            self.net.reservoir.W_u.weight = m_dict['reservoir.W_u.weight']

        # getting number of elements of every parameter
        self.n_params = {}
        self.train_params = []
        logging.info('Training the following parameters:')
        for k,v in self.net.named_parameters():
            # k is name, v is weight

            # filtering just for the parts that will be trained
            for part in self.args.train_parts:
                if part in k:
                    logging.info(f'  {k}')
                    self.n_params[k] = (v.shape, v.numel())
                    self.train_params.append(v)

        self.criterion = get_criterion(self.args)
        
        self.optimizer = get_optimizer(self.args, self.train_params)

        self.dset = load_rb(self.args.dataset)
        
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

            if 'seq-goals' in self.args.dataset:
                targets = torch.Tensor(self.dset)
            else:
                dset = np.asarray([x[:-1] for x in self.dset])
                x = torch.from_numpy(dset[:,0,:]).float()
                y = torch.from_numpy(dset[:,1,:]).float()

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

                # need to do this so that burn in works
                # res state starting from same random seed for each iteration
                self.net.reset(res_state_seed=0)
                self.net.zero_grad()
                total_loss = torch.tensor(0.)


                # testing with my random loss function
                if 'seq-goals' in self.args.dataset:
                    n_pts = len(self.dset[0])
                    cur_indices = [0 for i in range(len(self.dset))]
                    for j in range(100):
                        net_in = targets[torch.arange(len(self.dset)),cur_indices,:]
                        action, next_state = self.net(net_in)
                        dists = []
                        done = 0
                        for seq in range(len(self.dset)):
                            dist = torch.norm(next_state[seq] - net_in[seq])
                            if dist < 0.1 and cur_indices[seq] < n_pts:
                                cur_indices[seq] += 1
                                done += 1
                            dists.append(dist)
                        # step_loss = sum([(n_pts + 1 - i) + dists[i] for i in cur_indices])
                        # giving a reward for being done
                        step_loss = sum(dists) - done
                        total_loss += step_loss

                else:
                    for j in range(x.shape[1]):
                        net_in = x[:,j].reshape(-1, self.args.L)
                        net_out = self.net(net_in)
                        net_target = y[:,j].reshape(-1, self.args.Z)

                        step_loss = self.criterion(net_out, net_target)
                        total_loss += step_loss

                total_loss.backward()

                grad_list = []
                for k,v in self.net.named_parameters():
                    for part in self.args.train_parts:
                        if part in k:
                            grad = v.grad.numpy().reshape(-1)
                            grad_list.append(grad)

                vec = np.concatenate(grad_list)
                post = np.float64(vec)

                return total_loss.item(), post

            # callback just does logging
            def callback(xk):
                if self.args.no_log:
                    return
                self.scipy_ix += 1
                if self.scipy_ix % self.log_interval == 0:
                    # W_f, W_ro = vec_to_param(xk)
                    sample_n = random.randrange(len(self.dset))

                    self.net.reset(res_state_seed=0)
                    self.net.zero_grad()
                    outs = []
                    total_loss = torch.tensor(0.)


                    if 'seq-goals' in self.args.dataset:
                        done = []
                        ins = []
                        n_pts = len(self.dset[0])
                        cur_index = 0
                        for j in range(100):
                            net_in = targets[sample_n,cur_index,:]
                            action, next_state = self.net(net_in)
                            dist = torch.norm(next_state - net_in)
                            ins.append(net_in.numpy())
                            outs.append(net_out.detach().squeeze().numpy())

                            done = 0
                            if dist < 0.1 and cur_index < n_pts :
                                cur_index += 1
                                done += 1
                                # if cur_index >= n_pts:
                                #     dist = 0
                                #     done 
                                # else:
                                #     cur_index = targets[sample_n,cur_index,:]
                            # dists.append(dist)
                            # step_loss = sum([(len(self.dset) - i) * dists[i] for i in cur_indices])
                            # step_loss = dist * (n_pts - cur_index + 1)
                            step_loss = dist - done
                            total_loss += step_loss
                        pdb.set_trace()
                        self.log_checkpoint(self.scipy_ix, np.array(ins), np.array(ins), np.array(outs), total_loss.item(), total_loss.item())

                    else:
                        xs = x[sample_n,:]
                        ys = y[sample_n,:]
                        for j in range(xs.shape[0]):
                            net_in = xs[j].reshape(-1, self.args.L)
                            net_out = self.net(net_in)
                            outs.append(net_out.item())
                            net_target = ys[j].reshape(-1, self.args.Z)
                            step_loss = self.criterion(net_out, net_target)
                            total_loss += step_loss

                        z = np.stack(outs).squeeze()
                        self.log_checkpoint(self.scipy_ix, xs.numpy(), ys.numpy(), z, total_loss.item(), total_loss.item())

                    logging.info(f'iteration {self.scipy_ix}\t| loss {total_loss.item():.3f}')

            # getting the initial values to put into the algorithm
            init_list = []
            for k,_ in self.n_params.items():
                init = self.net.state_dict()[k].numpy().reshape(-1)
                init_list.append(init)

            init = np.concatenate(init_list)
            optim_options = {
                'iprint': self.log_interval,
                'maxiter': self.args.maxiter,
                'ftol': 1e-12
            }
            optim = minimize(closure, init, method='L-BFGS-B', jac=True, callback=callback, options=optim_options)

            # W_f_final, W_ro_final = vec_to_param(optim.x)
            error_final = optim.fun
            n_iters = optim.nit
            # error_final = self.test()
            

            if not self.args.no_log:
                self.log_model(ix='final')
                with open(os.path.join(self.log.run_dir, f'checkpoints_{self.run_id}.pkl'), 'wb') as f:
                    pickle.dump(self.vis_samples, f)
                self.csv_path.close()

            return error_final, n_iters

    def log_model(self, ix=0):
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
        self.vis_samples.append([ix, x, y, z, total_loss, avg_loss])
        if os.path.exists(self.plot_checkpoint_path):
            os.remove(self.plot_checkpoint_path)
        with open(self.plot_checkpoint_path, 'wb') as f:
            pickle.dump(self.vis_samples, f)

    def train_iteration(self, x, y):
        self.net.reset()
        self.optimizer.zero_grad()

        
        outs = []
        sublosses = []
        total_loss = torch.tensor(0.)


        if 'seq-goals' in self.args.dataset:
            ins = []
            goals = []
            n_pts = x.shape[1]
            n_trials = x.shape[0]
            cur_indices = [0 for i in range(n_trials)]
            for j in range(100):
                net_in = x[torch.arange(n_trials),cur_indices,:]
                ins.append(net_in)
                goals.append(net_in)
                net_out, step_loss, extras = self.run_iteration(net_in, net_in)
                dists = torch.norm(net_out - net_in, dim=1)
                done = 0
                for seq in range(n_trials):
                    dist = dists[seq]
                    if dist < 0.1 and cur_indices[seq] < n_pts:
                        cur_indices[seq] += 1
                        done += 1
                # step_loss = sum([(n_pts + 1 - i) + dists[i] for i in cur_indices])
                # step_loss = sum(dists)# - done
                outs.append(net_out[-1].detach().numpy())
                total_loss += step_loss

            ins = torch.cat(ins)
            goals = torch.cat(goals)

        else:

            ins = x
            goals = y

            for j in range(x.shape[1]):
                # run the step

                net_out, step_loss, extras = self.run_iteration(x[:,j], y[:,j])

                sublosses.append(step_loss.item())
                if np.isnan(step_loss.item()):
                    return -1, (net_out, extras)
                total_loss += step_loss

                outs.append(net_out[-1].item())
                # thals.append(list(thal[-1].detach().numpy().squeeze()))
                # ress.append(list(res[-1].detach().numpy().squeeze()))

        total_loss.backward()

        return total_loss, (ins, goals, outs, None, None, None)
        return total_loss, (x, y, outs, thals, ress, sublosses)

    def run_iteration(self, x, y):
        net_in = x.reshape(-1, self.args.L)
        net_out, extras = self.net(net_in, extras=True)
        net_target = y.reshape(-1, self.args.Z)
        if 'seq-goals' in self.args.dataset:
            # calculate norm across batches
            dists = torch.norm(net_out - net_target, dim=1)
            step_loss = torch.sum(dists)
        else:
            step_loss = self.criterion(net_out, net_target)

        return net_out, step_loss, extras

        # if 'seq-goals' in self.args.dataset:

        #     net_in = x.reshape(-1, self.args.L)
        #     net_out, extras = self.net(net_in, extras=True)
        #     net_target = y





        #     n_pts = len(self.dset[0])
        #     cur_indices = [0 for i in range(len(self.dset))]
        #     for j in range(100):
        #         net_in = targets[torch.arange(len(self.dset)),cur_indices,:]
        #         net_out = self.net(net_in)
        #         dists = []
        #         done = 0
        #         for seq in range(len(self.dset)):
        #             dist = torch.norm(net_out[seq] - net_in[seq])
        #             if dist < 0.1 and cur_indices[seq] < n_pts:
        #                 cur_indices[seq] += 1
        #                 done += 1
        #             dists.append(dist)
        #         # step_loss = sum([(n_pts + 1 - i) + dists[i] for i in cur_indices])
        #         step_loss = sum(dists)# - done
        #         total_loss += step_loss


    def test(self, n=0):
        if n != 0:
            assert n <= len(self.dset)
            batch = np.random.choice(self.dset, n)
        else:
            batch = self.dset
        if 'seq-goals' in self.args.dataset:
            x = torch.Tensor(batch)
            y = x
        else:
            x, y, _ = list(zip(*batch))
            x = torch.Tensor(x)
            y = torch.Tensor(y)

        with torch.no_grad():
            self.net.reset()

            total_loss = torch.tensor(0.)
            if 'seq-goals' in self.args.dataset:
                n_pts = x.shape[1]
                n_trials = x.shape[0]
                cur_indices = [0 for i in range(n_trials)]
                for j in range(100):
                    net_in = x[torch.arange(n_trials),cur_indices,:]
                    net_out, step_loss, _ = self.run_iteration(net_in, net_in)
                    dists = torch.norm(net_out - net_in, dim=1)
                    done = 0
                    for seq in range(n_trials):
                        dist = dists[seq]
                        if dist < 0.1 and cur_indices[seq] < n_pts:
                            cur_indices[seq] += 1
                            done += 1
                    total_loss += step_loss

            else:
                for j in range(x.shape[1]):
                    _, step_loss, _ = self.run_iteration(x[:,j], y[:,j])
                    total_loss += step_loss

        return total_loss.item()

    def train(self, ix_callback=None):
        ix = 0

        its_p_epoch = len(self.dset) // self.args.batch_size
        logging.info(f'Dataset size {len(self.dset)} | batch size {self.args.batch_size} --> {its_p_epoch} iterations / epoch')

        # for convergence testing
        max_abs_grads = []
        running_min_error = float('inf')
        running_no_min = 0

        running_loss = 0.0
        running_mag = 0.0
        ending = False
        for e in range(self.args.n_epochs):
            np.random.shuffle(self.dset)
            epoch_idx = 0
            while epoch_idx < its_p_epoch:
                epoch_idx += 1
                batch = self.dset[(epoch_idx-1) * self.args.batch_size:epoch_idx * self.args.batch_size]
                if len(batch) < self.args.batch_size:
                    break
                ix += 1
                if 'seq-goals' in self.args.dataset:
                    x = torch.Tensor(batch)
                    y = x
                else:
                    x, y, _ = list(zip(*batch))
                    x = torch.Tensor(x)
                    y = torch.Tensor(y)

                loss, etc = self.train_iteration(x, y)
                if ix_callback is not None:
                    ix_callback(loss, etc)
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
                    test_loss = self.test()
                    avg_max_grad = running_mag / self.log_interval
                    logging.info(f'iteration {ix}\t| loss {avg_loss:.3f}\t| max abs grad {avg_max_grad:.3f} | test loss {test_loss:.3f}')

                    if not self.args.no_log:
                        self.log_checkpoint(ix, etc[0].numpy(), etc[1].numpy(), z, running_loss, avg_loss)
                    running_loss = 0.0
                    running_mag = 0.0

                    # convergence based on no avg loss decrease after patience samples
                    if self.args.conv_type == 'patience':
                        if test_loss < running_min_error:
                            running_no_min = 0
                            running_min_error = test_loss
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
            logging.info(f'Finished dataset epoch {e+1}')
            if ending:
                break

        if not self.args.no_log:
            # for later visualization of outputs over timesteps
            with open(os.path.join(self.log.run_dir, f'checkpoints_{self.run_id}.pkl'), 'wb') as f:
                pickle.dump(self.vis_samples, f)

            self.csv_path.close()

        final_loss = self.test()
        logging.info(f'END | iterations: {(ix // self.log_interval) * self.log_interval} | test loss: {final_loss}')
        return final_loss, ix

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-L', type=int, default=1, help='latent input dimension')
    parser.add_argument('-T', type=int, default=1, help='task dimension')
    parser.add_argument('-D', type=int, default=5, help='intermediate dimension')
    parser.add_argument('-N', type=int, default=50, help='number of neurons in reservoir')
    parser.add_argument('-Z', type=int, default=1, help='output dimension')

    parser.add_argument('--net', type=str, default='basic', choices=['basic', 'state'])

    parser.add_argument('--train_parts', type=str, nargs='+', default=['W_ro', 'W_f'])
    parser.add_argument('--stride', type=int, default=1, help='stride of the W_f')
    
    # make sure model_config path is specified if you use any paths! it ensures correct dimensions, bias, etc.
    parser.add_argument('--model_config_path', type=str, default=None, help='config path corresponding to model load path')
    parser.add_argument('--model_path', type=str, default=None, help='start training from certain model. superseded by below')
    parser.add_argument('--Wro_path', type=str, default=None, help='start training from certain Wro')
    parser.add_argument('--Wf_path', type=str, default=None, help='start training from certain Wf')
    parser.add_argument('--reservoir_path', type=str, default=None, help='saved reservoir. should be saved with seed tho')
    
    parser.add_argument('--res_init_type', type=str, default='gaussian', help='')
    parser.add_argument('--res_init_gaussian_std', type=float, default=1.5)
    parser.add_argument('--network_delay', type=int, default=0)
    parser.add_argument('--reservoir_noise', type=float, default=0)
    parser.add_argument('--no_bias', action='store_true')
    parser.add_argument('--out_act', type=str, default=None, help='output activation')

    parser.add_argument('--dataset', type=str, default='datasets/rsg2.pkl')

    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'rmsprop', 'lbfgs-scipy', 'lbfgs-pytorch'], default='lbfgs-scipy')
    parser.add_argument('--loss', type=str, default='mse')

    # lbfgs-scipy arguments
    parser.add_argument('--maxiter', type=int, default=10000, help='limit to # iterations. lbfgs-scipy only')

    # adam arguments
    parser.add_argument('--batch_size', type=int, default=1, help='size of minibatch used')
    parser.add_argument('--conv_type', type=str, choices=['patience', 'grad'], default='patience', help='how to determine convergence. adam only')
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
    parser.add_argument('--log_checkpoint_models', action='store_true')

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
    if args.reservoir_seed is None:
        args.reservoir_seed = random.randrange(1e6)
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
    if args.model_path is not None and args.model_config_path is not None:
        with open(args.model_config_path) as f:
            config = json.load(f)
        args.N = config['N']
        args.D = config['D']
        args.L = config['L']
        args.Z = config['Z']
        args.bias = config['bias']
        args.reservoir_seed = config['reservoir_seed']

    # shortcut for specifying train everything including reservoir
    if args.train_parts == ['all']:
        args.train_parts = ['W_ro', 'W_f', 'reservoir']

    # output activation depends on the task / dataset used
    if args.out_act is None:
        if 'rsg' in args.dataset:
            args.out_act = 'exp'
        else:
            args.out_act = 'none'

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
    if args.optimizer == 'lbfgs-scipy':
        final_loss, n_iters = trainer.optimize_lbfgs('scipy')
    elif args.optimizer == 'lbfgs-pytorch':
        final_loss, n_iters = trainer.optimize_lbfgs('pytorch')
    elif args.optimizer in ['sgd', 'rmsprop', 'adam']:
        final_loss, n_iters = trainer.train()

    if args.slurm_id is not None:
        # if running many jobs, then we gonna put the results into a csv
        csv_path = os.path.join('logs', args.name.split('_')[0] + '.csv')
        csv_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            labels_csv = ['slurm_id', 'D', 'N', 'bias', 'seed', 'rseed', 'xseed', 'rnoise', 'dset', 'niter', 'loss']
            vals_csv = [
                args.slurm_id, args.D, args.N, args.bias, args.seed,
                args.reservoir_seed, args.reservoir_x_seed, args.reservoir_noise,
                args.dataset, n_iters, final_loss
            ]
            if args.optimizer == 'adam':
                labels_csv.extend(['lr', 'epochs'])
                vals_csv.extend([args.lr, args.n_epochs])
            elif args.optimizer == 'lbfgs-scipy':
                pass

            if not csv_exists:
                writer.writerow(labels_csv)
            writer.writerow(vals_csv)

    logging.shutdown()



