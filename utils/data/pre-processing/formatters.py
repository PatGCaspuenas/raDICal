import numpy as np
import h5py
import scipy.io as sio
import os
import pandas as pd

# NOT PART OF THE CODE - pre-processing of data output by MATLAB's GetDatasets.m

def mat2hdf5_FP_flow(path_flow, path_save):

    M_flow = sio.loadmat(path_flow)

    X = M_flow['X']
    Y = M_flow['Y']

    U = M_flow['u']
    V = M_flow['v']
    #dUdt = M_flow['du']
    #dVdt = M_flow['dv']
    #p = M_flow['p']

    t = (M_flow['Nimg']-1).T*0.1

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(U)[1]
    Re = 150

    # Body mask
    R = 0.5

    xF, yF = -3/2 * np.cos(30 * np.pi / 180), 0
    xB, yB = 0, -3/4
    xT, yT = 0, 3/4

    B = np.zeros((m, n))
    B[:, :] = 'nan'
    B[np.where(R ** 2 - (X - xF) ** 2 - (Y - yF) ** 2 >= 0)] = 1
    B[np.where(R ** 2 - (X - xB) ** 2 - (Y - yB) ** 2 >= 0)] = 1
    B[np.where(R ** 2 - (X - xT) ** 2 - (Y - yT) ** 2 >= 0)] = 1

    U[np.reshape(B, (m * n), order='F') == 1, :] = 0
    V[np.reshape(B, (m * n), order='F') == 1, :] = 0
    #dUdt[np.reshape(B, (m * n), order='F') == 1, :] = 0
    #dVdt[np.reshape(B, (m * n), order='F') == 1, :] = 0
    #p[np.reshape(B, (m * n), order='F') == 1, :] = 0

    U = np.reshape(U, (m * n, nt), order='F')
    V = np.reshape(V, (m * n, nt), order='F')
    #dUdt = np.reshape(dUdt, (m * n, nt), order='F')
    #dVdt = np.reshape(dVdt, (m * n, nt), order='F')
    #p = np.reshape(p, (m * n, nt), order='F')

    #flow = {'U': U, 'V': V, 'dUdt': dUdt, 'dVdt': dVdt, 'P':p, 't': t, 'Re': Re}
    flow = {'U': U, 'V': V, 't': t, 'Re': Re}

    with h5py.File(path_save, 'w') as h5file:
        for key, item in flow.items():
            h5file.create_dataset(key, data=item)

def mat2hdf5_SC(path_grid, path_flow, path_grid_save, path_flow_save):

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

    Re = 100

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
    flow = {'U': U, 'V': V, 'W': W, 't': t, 'Re': Re}

    with h5py.File(path_flow_save, 'w') as h5file:
        for key, item in flow.items():
            h5file.create_dataset(key, data=item)

    with h5py.File(path_grid_save, 'w') as h5file:
        for key, item in grid.items():
            h5file.create_dataset(key, data=item)


def add_control_FP(path_input, path_save):

    M_input = sio.loadmat(path_input)

    vF = M_input['vF']
    vT = M_input['vT']
    vB = M_input['vB']

    with h5py.File(path_save, 'a') as h5file:
        h5file.create_dataset('vF', data=vF)
        h5file.create_dataset('vT', data=vT)
        h5file.create_dataset('vB', data=vB)

def mat2hdf5_FP_grid(path, path_save):

    M = sio.loadmat(path)

    X = M['X']
    Y = M['Y']

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    # Body mask
    R = 0.5

    xF, yF = -3 / 2 * np.cos(30 * np.pi / 180), 0
    xB, yB = 0, -3 / 4
    xT, yT = 0, 3 / 4

    B = np.zeros((m, n))
    B[:, :] = 'nan'
    B[np.where(R ** 2 - (X - xF) ** 2 - (Y - yF) ** 2 >= 0)] = 1
    B[np.where(R ** 2 - (X - xB) ** 2 - (Y - yB) ** 2 >= 0)] = 1
    B[np.where(R ** 2 - (X - xT) ** 2 - (Y - yT) ** 2 >= 0)] = 1

    grid = {'X': X, 'Y': Y, 'B': B}

    with h5py.File(path_save, 'w') as h5file:
        for key, item in grid.items():
            h5file.create_dataset(key, data=item)

