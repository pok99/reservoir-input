import numpy as np
import torch
import matplotlib.pyplot as plt

import pdb
import sys
import random

from network import Network, Reservoir
from utils import Bunch

b = Bunch(N=20, D=1, res_init_params={'std':1.5}, reservoir_noise=0.1, reservoir_burn_steps=200)

net = Network(b)

N = net.args.N
D = net.args.D

corrs = []
all_dists = []

n_steps = 300
n_net_reps = 12
n_reps = 15

for rep in range(n_net_reps):

    net = Network(b)
    init_x = np.random.normal(0, 1, (1, N))
    net.reservoir.reset(res_state=init_x)
    #x1 = net.reservoir.x.numpy().reshape(-1)
    xs = np.zeros((n_steps, N))
    for i in range(n_steps):
        net(torch.zeros(1))
        xs[i] = net.reservoir.x.detach().numpy().reshape(-1)

    dists = []

    for i in range(n_reps):
        new_x = init_x + np.random.normal(0, .1, (1, N))
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

    # plt.xlabel('steps since burn in')
    # plt.ylabel('distance')
plt.show()


    #print('i', x1, '\nn', x2)

#     corr = np.dot(x1/np.linalg.norm(x1), x2/np.linalg.norm(x2))
#     dist = np.linalg.norm(x1 - x2)
#     #print(corr)
#     corrs.append(corr)
#     dists.append(dist)

# plt.subplot(2, 1, 1)
# plt.hist(corrs, bins=20, range=[-1,1])

# plt.subplot(2, 1, 2)
# plt.hist(dists, bins=30, alpha=0.5)

# plt.tight_layout()
# plt.show()