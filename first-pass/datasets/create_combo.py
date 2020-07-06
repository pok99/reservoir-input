import numpy as np
import sys
import pdb
import pickle

sys.path.append('../')
from dataset import load_dataset, save_dataset

dsets = [
    'rsg_s10.pkl',
    'rsg_s15.pkl',
    'rsg_s20.pkl',
    'rsg_s25.pkl',
    'rsg_s30.pkl',
    'rsg_s35.pkl',
    'rsg_s40.pkl'
]

new_set = []

for d in dsets:
    dset = load_dataset(d)
    np.random.shuffle(dset)

    new_set.append(dset[:143])

new_set = [x for y in new_set for x in y][:-1]
assert len(new_set) == 1000


with open('rsg_s10-5-40.pkl', 'wb') as f:
    pickle.dump(new_set, f)