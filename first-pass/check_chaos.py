import numpy as np
import torch
import matplotlib.pyplot as plt

import pdb
import sys
import random

from network import Network, Reservoir
from utils import Bunch

b = Bunch(N=20, D=1, res_init_params={'std':1.5}, reservoir_noise=.1, reservoir_burn_steps=200)

net = Network(b)

N = net.args.N
D = net.args.D

corrs = []
all_dists = []

n_steps = 300
n_net_reps = 12
n_reps = 15
activation = lambda y: y

for rep in range(n_net_reps):

    net = Network(b)
    net.reservoir.activation = activation
    init_x = np.random.normal(0, 1, (1, N))
    net.reservoir.reset(res_state=init_x)
    #x1 = net.reservoir.x.numpy().reshape(-1)
    xs = np.zeros((n_steps, N))
    for i in range(n_steps):
        net(torch.zeros(1))
        xs[i] = net.reservoir.x.detach().numpy().reshape(-1)

    dists = []

    for i in range(n_reps):
        new_x = init_x + np.random.normal(0, 1, (1, N))
        net.reservoir.reset(res_state=new_x)
        #x2 = net.reservoir.x.detach().numpy().reshape(-1)
        xss = np.zeros((n_steps, N))
        dist = np.zeros(n_steps)
        for j in range(n_steps):
            net(torch.zeros(1))
            xss[j] = net.reservoir.x.detach().numpy().reshape(-1)
            dist[j] = np.linalg.norm(xss[j] - xs[j])
        dists.append(dist)

    all_dists.append(dists)



for i in range(n_net_reps):
    plt.subplot(3, 4, i+1)
    for j in range(n_reps):
        plt.plot(all_dists[i][j])

plt.show()
