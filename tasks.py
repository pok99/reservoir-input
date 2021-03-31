import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import pickle
import os
import sys
import json
import pdb
import random
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import collections as matcoll

import argparse

# from motifs import gen_fn_motifs
from utils import update_args, get_file_args, load_rb, Bunch

eps = 1e-6

mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.linewidth'] = .5

c_order = ['coral', 'cornflowerblue', 'forestgreen', 'orchid']


class Task:
    def __init__(self, t_len, dset_id=None, n=None):
        self.t_len = t_len
        self.dset_id = dset_id
        self.n = n

    def get_x(self):
        pass

    def get_y(self):
        pass

class RSG(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        if args.intervals is None:
            t_p = np.random.randint(args.min_t, args.max_t)
        else:
            t_p = random.choice(args.intervals)
        ready_time = np.random.randint(args.p_len * 2, args.max_ready)
        set_time = ready_time + t_p
        go_time = set_time + t_p
        assert go_time < args.t_len

        self.t_type = args.t_type
        self.p_len = args.p_len
        self.rsg = (ready_time, set_time, go_time)
        self.t_p = t_p

    def get_x(self, args=None):
        x = np.zeros((5, self.t_len))
        rt, st, gt = self.rsg
        # ready pulse
        x[0, rt:rt+self.p_len] = 1
        # set pulse
        x_set = np.zeros(self.t_len)
        x_set[st:st+self.p_len] = 1
        # perceptual shift
        if args is not None and args.m_noise != 0:
            x_set = shift_x(x_set, trial)
        # are ready/set different signals?
        if args is not None and args.separate_signal:
            x[1] = x_set
        else:
            x[0] += x_set
        if args is not None and args.x_noise != 0:
            x = corrupt_x(args, x)
        return x

    def get_y(self, args=None):
        y = np.arange(self.t_len)
        slope = 1 / self.t_p
        y = y * slope - self.rsg[1] * slope
        y = np.clip(y, 0, 1.5)
        return y

class CSG(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        if args.intervals is None:
            t_p = np.random.randint(args.min_t, args.max_t)
            t_percentile = (t_p - args.min_t) / (args.max_t - args.min_t)
        else:
            ix = np.random.randint(len(args.intervals))
            t_p = args.intervals[ix]
            t_percentile = ix / len(args.intervals)
        cue_time = np.random.randint(args.p_len * 2, args.max_cue)
        set_time = cue_time + np.random.randint(args.p_len * 2, args.max_cue)
        go_time = set_time + t_p
        assert go_time < self.t_len

        self.t_type = args.t_type
        self.p_len = args.p_len
        self.t_percentile = t_percentile
        self.csg = (cue_time, set_time, go_time)
        self.t_p = t_p

    def get_x(self, args=None):
        x = np.zeros((5, self.t_len))
        ct, st, gt = self.csg
        x[0, ct:ct+self.p_len] = self.t_percentile
        x[0, st:st+self.p_len] = 1
        return x

    def get_y(self, args=None):
        y = np.arange(self.t_len)
        slope = 1 / self.t_p
        y = y * slope - self.csg[1] * slope
        y = np.clip(y, 0, 1.5)
        return y

class DelayProAnti(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        theta = np.random.random() * 2 * np.pi
        stimulus = [np.cos(theta), np.sin(theta)]

        self.t_type = args.t_type
        assert self.t_type in ['delay-pro', 'delay-anti']
        self.stimulus = stimulus
        self.fix = args.fix_t
        self.stim = self.fix + args.stim_t

    def get_x(self, args=None):
        x = np.zeros((5, self.t_len))
        # 0 is fixation, the remainder are stimulus
        x[0,:self.stim] = 1
        x[1,self.fix:] = self.stimulus[0]
        x[2,self.fix:] = self.stimulus[1]
        return x

    def get_y(self, args=None):
        y = np.zeros((5, self.t_len))
        y[0,:self.stim] = 1
        y[1,self.stim:] = self.stimulus[0]
        y[2,self.stim:] = self.stimulus[1]
        if self.t_type.endswith('anti'):
            y[1:,] = -y[1:,]
        return y

class MemoryProAnti(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)
        theta = np.random.random() * 2 * np.pi
        stimulus = [np.cos(theta), np.sin(theta)]

        self.t_type = args.t_type
        assert self.t_type in ['memory-pro', 'memory-anti']
        self.stimulus = stimulus
        self.fix = args.fix_t
        self.stim = self.fix + args.stim_t
        self.memory = self.stim + args.memory_t

    def get_x(self, args=None):
        x = np.zeros((5, self.t_len))
        x[0,:self.memory] = 1
        x[1,self.fix:self.stim] = self.stimulus[0]
        x[2,self.fix:self.stim] = self.stimulus[1]
        return x

    def get_y(self, args=None):
        y = np.zeros((5, self.t_len))
        y[0,:self.memory] = 1
        y[1,self.memory:] = self.stimulus[0]
        y[2,self.memory:] = self.stimulus[1]
        # reversing output stimulus for anti condition
        if self.t_type.endswith('anti'):
            y[1:,] = -y[1:,]
        return y

class DelayCopy(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)

        self.s_len = self.t_len // 2
        x_r = np.arange(self.s_len)

        x = np.zeros((args.dim, self.s_len))
            
        freqs = np.random.uniform(args.f_range[0], args.f_range[1], (args.dim, args.n_freqs))
        amps = np.random.uniform(-args.amp, args.amp, (args.dim, args.n_freqs))

        for i in range(args.dim):
            for j in range(args.n_freqs):
                x[i] = x[i] + amps[i,j] * np.sin(1/freqs[i,j] * x_r) / np.sqrt(args.n_freqs)

        self.t_type = args.t_type
        self.dim = args.dim
        self.pattern = x

    def get_x(self, args=None):
        x = np.zeros((5, self.t_len))
        x[:self.dim, :self.s_len] = self.pattern
        return x

    def get_y(self, args=None):
        y = np.zeros((5, self.t_len))
        y[:self.dim, self.s_len:] = self.pattern
        return y

class FlipFlop(Task):
    def __init__(self, args, dset_id=None, n=None):
        super().__init__(args.t_len, dset_id, n)

        keys = []
        for i in range(args.dim):
            cum_xlen = 0
            # add new dimension
            keys.append([])
            while cum_xlen < self.t_len:
                cum_xlen += np.random.geometric(args.geop) + args.p_len
                if cum_xlen < self.t_len:
                    sign = np.random.choice([-1, 1])
                    keys[i].append(sign * (cum_xlen - args.p_len))

        self.t_type = args.t_type
        self.p_len = args.p_len
        self.dim = args.dim
        self.keys = keys

    def get_x(self, args=None):
        x = np.zeros((self.dim, self.t_len))
        for i in range(self.dim):
            for idx in self.keys[i]:
                x[i, abs(idx):abs(idx)+self.p_len] = np.sign(idx)
        return x

    def get_y(self, args=None):
        y = np.zeros((self.dim, self.t_len))
        for i in range(self.dim):
            for j in range(len(self.keys[i])):
                # the desired key we care about
                idx = self.keys[i][j]
                # the sign to assign to this one
                sign = np.sign(idx)
                if j == len(self.keys[i]) - 1:
                    y[i, np.abs(idx):] = sign
                else:
                    idxs = np.abs(self.keys[i][j:j+2])
                    y[i, idxs[0]:idxs[1]] = sign
        return y

# ways to add noise to x
def corrupt_x(args, x):
    x += np.random.normal(scale=args.x_noise, size=x.shape)
    return x

def shift_x(args, x, trial):
    if args.m_noise == 0:
        return x
    t_p = trial['t_p']
    disp = int(np.random.normal(0, args.m_noise*t_p/50))
    x = x.roll(disp)
    return x

def create_dataset(args):
    t_type = args.t_type
    n_trials = args.n_trials

    if t_type.startswith('rsg'):
        TaskObj = RSG
    elif t_type.startswith('csg'):
        TaskObj = CSG
    elif t_type == 'delay-copy':
        TaskObj = DelayCopy
    elif t_type == 'flip-flop':
        TaskObj = FlipFlop
    elif t_type == 'delay-pro' or t_type == 'delay-anti':
        assert args.fix_t + args.stim_t < args.t_len
        TaskObj = DelayProAnti
    elif t_type == 'memory-pro' or t_type == 'memory-anti':
        assert args.fix_t + args.stim_t + args.memory_t < args.t_len
        TaskObj = MemoryProAnti
    else:
        raise NotImplementedError

    trials = []
    for n in range(n_trials):
        trial = TaskObj(args, args.name, n)
        trials.append(trial)

    return trials, args

# turn task_args argument into usable argument variables
def get_task_args(args):
    tarr = args.task_args
    targs = Bunch()
    if args.t_type.startswith('rsg'):
        targs.p_len = get_tval(tarr, 'pl', 5, int)
        targs.max_ready = get_tval(tarr, 'max_ready', 80, int)
        if args.intervals is None:
            targs.min_t = get_tval(tarr, 'gt', targs.p_len * 4, int)
            targs.max_t = get_tval(tarr, 'lt', args.t_len // 2 - targs.p_len * 4, int)

    elif args.t_type.startswith('csg'):
        targs.p_len = get_tval(tarr, 'pl', 5, int)
        targs.max_cue = get_tval(tarr, 'max_cue', 80, int)
        targs.max_set = get_tval(tarr, 'max_set', 200, int)
        if args.intervals is None:
            targs.min_t = get_tval(tarr, 'gt', targs.p_len * 4, int)
            targs.max_t = get_tval(tarr, 'lt', args.t_len // 2 - targs.p_len * 4, int)

    elif args.t_type == 'delay-copy':
        targs.dim = get_tval(tarr, 'dim', 2, int)
        targs.n_freqs = get_tval(tarr, 'n_freqs', 20, int)
        targs.f_range = get_tval(tarr, 'f_range', [10, 40], float, n_vals=2)
        targs.amp = get_tval(tarr, 'amp', 1, float)

    elif args.t_type == 'flip-flop':
        targs.dim = get_tval(tarr, 'dim', 3, int)
        targs.p_len = get_tval(tarr, 'pl', 5, int)
        targs.geop = get_tval(tarr, 'p', .02, float)

    elif args.t_type == 'delay-pro' or args.t_type == 'delay-anti':
        targs.fix_t = get_tval(tarr, 'fix', 50, int)
        targs.stim_t = get_tval(tarr, 'stim', 100, int)

    elif args.t_type == 'memory-pro' or args.t_type == 'memory-anti':
        targs.fix_t = get_tval(tarr, 'fix', 50, int)
        targs.stim_t = get_tval(tarr, 'stim', 100, int)
        targs.memory_t = get_tval(tarr, 'memory', 50, int)

    return targs

# get particular value(s) given name and casting type
def get_tval(targs, name, default, dtype, n_vals=1):
    if name in targs:
        idx = targs.index(name)
        if n_vals == 1:
            val = dtype(targs[idx + 1])
        else:
            vals = []
            for i in range(1, n_vals+1):
                vals.append(dtype(targs[idx + i]))
    else:
        val = default
    return val


def save_dataset(dset, name, config=None):
    fname = os.path.join('datasets', name + '.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(dset, f)
    gname = os.path.join('datasets', 'configs', name + '.json')
    if config is not None:
        with open(gname, 'w') as f:
            json.dump(config.to_json(), f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='load', choices=['create', 'load'])
    parser.add_argument('name')
    parser.add_argument('-c', '--config', default=None, help='create from a config file')

    # general dataset arguments
    parser.add_argument('-t', '--t_type', default='rsg', help='type of trial to create')
    parser.add_argument('-l', '--t_len', type=int, default=500)
    parser.add_argument('-n', '--n_trials', type=int, default=2000)

    # rsg arguments
    parser.add_argument('-i', '--intervals', nargs='*', type=int, default=None, help='select from rsg intervals')

    parser.add_argument('--motifs', type=str, help='path to motifs')
    parser.add_argument('-a', '--task_args', nargs='*', default=[], help='terms to specify parameters of trial type')

    
    args = parser.parse_args()
    config_args = get_file_args(args.config)
    task_args = get_task_args(args)
    args = update_args(args, config_args)
    args = update_args(args, task_args)

    args.argv = ' '.join(sys.argv)

    if args.mode == 'create':
        dset, config = create_dataset(args)
        save_dataset(dset, args.name, config=config)
    elif args.mode == 'load':
        dset = load_rb(args.name)
        t_type = dset[0].t_type
        x_range = np.arange(dset[0].t_len)

        samples = random.sample(dset, 12)
        fig, ax = plt.subplots(3,4,sharex=True, sharey=True, figsize=(10,6))
        for i, ax in enumerate(fig.axes):
            ax.axvline(x=0, color='dimgray', alpha = 1)
            ax.axhline(y=0, color='dimgray', alpha = 1)
            ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
            ax.tick_params(axis='both', color='white')
            #ax.set_title(sample[i][2])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            trial = samples[i]
            trial_x = trial.get_x()
            trial_y = trial.get_y()

            if t_type.startswith('rsg') or t_type.startswith('csg'):
                trial_x = np.sum(trial_x, axis=0)
                ml, sl, bl = ax.stem(x_range, trial_x, use_line_collection=True, linefmt='coral', label='ready/set')
                ml.set_markerfacecolor('coral')
                ml.set_markeredgecolor('coral')
                if t_type == 'rsg-bin':
                    ml, sl, bl = ax.stem(x_range, [1], use_line_collection=True, linefmt='dodgerblue', label='go')
                    ml.set_markerfacecolor('dodgerblue')
                    ml.set_markeredgecolor('dodgerblue')
                else:
                    ax.plot(x_range, trial_y, color='dodgerblue', label='go', lw=2)

            elif t_type == 'delay-copy':
                for j in range(trial.dim):
                    ax.plot(x_range, trial_x[j], color=c_order[j], ls='--', lw=1)
                    ax.plot(x_range, trial_y[j], color=c_order[j], lw=1)

            elif t_type == 'flip-flop':
                for j in range(trial.dim):
                    ax.plot(x_range, trial_x[j], color=c_order[j], lw=.5)
                    ax.plot(x_range, trial_y[j], color=c_order[j], lw=1, ls='--', alpha=.9)

            elif t_type.endswith('pro') or t_type.endswith('anti'):
                ax.plot(x_range, trial_x[0], color='grey', lw=1, ls='--', alpha=.6)
                ax.plot(x_range, trial_x[1], color='salmon', lw=1, ls='--', alpha=.6)
                ax.plot(x_range, trial_x[2], color='dodgerblue', lw=1, ls='--', alpha=.6)
                ax.plot(x_range, trial_y[0], color='grey', lw=2)
                ax.plot(x_range, trial_y[1], color='salmon', lw=2)
                ax.plot(x_range, trial_y[2], color='dodgerblue', lw=2)

        handles, labels = ax.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='lower center')
        plt.show()