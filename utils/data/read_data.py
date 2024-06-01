import numpy as np
import scipy.io as sio
import h5py

def read_SC(path_grid, path_flow):


    M = sio.loadmat(path_grid)

    X = M['X']
    Y = M['Y']

    M = sio.loadmat(path_flow)

    U = M['u']
    V = M['v']
    W = M['w']

    t = M['t'].T

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(U)[1]

    # Body mask
    B = np.zeros((m ,n))
    B[np.where( 0.5**2 - X**2 - Y**2 >= 0)] = 1
    B[np.where(0.5 ** 2 - X ** 2 - Y ** 2 < 0)] = 'nan'

    U[np.reshape(B, (m*n), order='F') == 1, :] = 0
    V[np.reshape(B, (m*n), order='F') == 1, :] = 0
    W[np.reshape(B, (m*n), order='F') == 1, :] = 0

    U = np.reshape(U, (m * n, nt), order='F')
    V = np.reshape(V, (m * n, nt), order='F')
    W = np.reshape(W, (m * n, nt), order='F')

    grid = {'X': X, 'Y': Y, 'B': B}
    flow = {'U': U, 'V': V, 'W': W, 't': t}

    return grid, flow

def read_FP(path_grid, path_flow):

    grid = {}
    with h5py.File(path_grid, 'r') as f:
        for i in f.keys():
            grid[i] = f[i][()]

    flow = {}
    with h5py.File(path_flow, 'r') as f:
        for i in f.keys():
            flow[i] = f[i][()]

    return grid, flow

