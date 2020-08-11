import numpy as np
import torch

import matplotlib.pyplot as plt

import argparse
import pdb
import os

from network import Network
from utils import Bunch, load_rb


checkpoint_dir = 'logs/vanish_test/checkpoints_1753447'

files = ['model_100.pth', 'model_1000.pth', 'model_2000.pth']

W_fs = []
W_ros = []
for fn in files:
    path = os.path.join(checkpoint_dir, fn)
    with open(path, 'rb') as f:
        m_dict = torch.load(f)

        W_fs.append(m_dict['W_f.weight'])
        W_ros.append(m_dict['W_ro.weight'])

W_fs = np.stack(W_fs)
W_ros = np.stack(W_ros)
pdb.set_trace()