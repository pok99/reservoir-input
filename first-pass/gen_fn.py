import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler

mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:red', 'tab:blue', 'tab:purple'])

import pdb

from sklearn.gaussian_process import GaussianProcessRegressor as gpr

from utils import load_rb, lrange

eps = 1e-6

# create fn from motifs
def gen_fn_motifs(motifs, length=50, pause=5, t_var=.01, x_var=.1, smoothing=True):

    cur_y = 0
    
    y = np.array([])
    while len(y) < length:
        cm = np.random.choice(motifs)
        cm = cm[1:]
        if smoothing and len(y) > 0:
            x1 = len(y)-1
            cur_slope = cm[1]
            a = (cur_slope - prev_slope) / (2 * pause)
            b = prev_slope - 2 * a * x1
            c = y[-1] - a * x1 ** 2 - b * x1
            x = np.arange(len(y), len(y) + pause)
            y_x = a * x ** 2 + b * x + c
            y = np.concatenate((y, y_x))
            cur_y = y[-1]
        prev_slope = cm[-1] - cm[-2]
        y = np.concatenate((y, cm + cur_y))
        cur_y = y[-1]

    return y


def test_1d():

    plt.style.use('ggplot')
    fns = []
    motifs = load_rb('motifsets/temp.pkl')
    traj = gen_fn_motifs(motifs, length=200, smoothing=True)
    plt.plot(traj)
    
    plt.xlabel('timestep')
    plt.ylabel('activity')
    plt.show()


def test_3d():

    (xp, yp), (x, y) = gen_fn(dim=3, length=30, precision=0.1, interval=2, scale=2, return_all=True)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_pts = y[:,0]
    y_pts = y[:,1]
    z_pts = y[:,2]
    ax.scatter3D(yp[:,0], yp[:,1], yp[:,2], c='black', s=60)
    ax.scatter3D(x_pts, y_pts, z_pts, c=x, cmap='hsv', s=10)

    ax.grid(False)
    plt.axis('off')
    z100 = np.zeros(100)
    a100 = np.linspace(-5, 5, 100)
    ax.plot3D(z100, z100, a100, color='black')
    ax.plot3D(z100, a100, z100, color='black')
    ax.plot3D(a100, z100, z100, color='black')

    ax.set_xlim3d([-5, 5])
    ax.set_ylim3d([-5, 5])
    ax.set_zlim3d([-5, 5])

    plt.show()

def test_motifs():
    plt.style.use('ggplot')
    motifs, config = load_rb('motifsets/temp1d.pkl')

    fns = []
    traj = gen_fn_motifs(motifs, length=50)
    t, x = traj.tx()
    ta, xa = traj.anchors
    first = plt.scatter(ta, xa)
    plt.plot(t, x)

    plt.title('Example neural trajectories')

    #plt.ylim([-5, 5])
    plt.show()



if __name__ == '__main__':
    test_1d()
    #test_motifs()


