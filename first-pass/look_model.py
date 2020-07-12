import numpy as np
import torch
import matplotlib.pyplot as plt

import argparse
import pdb

from helpers import load_model
from reservoir import Network
from utils import Bunch


# parser = argparse.ArgumentParser()
# parser.add_argument('model')
# args = parser.parse_args()

# with open(args.model, 'rb') as f:
#     m_dict = torch.load(f)


# net = load_model(m_dict)




# bunch = Bunch()
# bunch.N = 250
# bunch.D = 250
# bunch.O = 1

# bunch.res_init_type = 'gaussian'
# bunch.res_init_params = {'std': 1.5}
# bunch.reservoir_seed = 0
# net = Network(bunch)

nums = ['146', '152', '158', '164', '170', '176', '182', '188']
#dsets = ['15', '25', '35', '45', '55', '65', '75', '85']
dsets = ['logs/2625461/'+i+'/model.pth' for i in nums]
models = []
for i in dsets:
    with open(i, 'rb') as f:
        models.append(torch.load(f))

pdb.set_trace()