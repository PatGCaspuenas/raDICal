import numpy as np
import h5py
import scipy.io as sio

def mat2hdf5_FP_flow(path_flow, path_grid, path_save):

    M_flow = sio.loadmat(path_flow)
    M_grid = sio.loadmat(path_grid)

    X = M_grid['X']
    Y = M_grid['Y']

    U = M_flow['u']
    V = M_flow['v']
    dUdt = M_flow['du']
    dVdt = M_flow['dv']

    t = M_flow['t'].T

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(U)[1]
    Re = 130

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
    dUdt[np.reshape(B, (m * n), order='F') == 1, :] = 0
    dVdt[np.reshape(B, (m * n), order='F') == 1, :] = 0

    U = np.reshape(U, (m * n, nt), order='F')
    V = np.reshape(V, (m * n, nt), order='F')
    dUdt = np.reshape(dUdt, (m * n, nt), order='F')
    dVdt = np.reshape(dVdt, (m * n, nt), order='F')

    flow = {'U': U, 'V': V, 'dUdt': dUdt, 'dVdt': dVdt, 't': t, 'Re': Re}

    with h5py.File(path_save, 'w') as h5file:
        for key, item in flow.items():
            h5file.create_dataset(key, data=item)


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



mat2hdf5_FP_flow(r'F:\AEs_wControl\utils\data\FP_14k_24k_AE.mat', r'F:\AEs_wControl\utils\data\FP_grid_AE.mat', \
                 r'F:\AEs_wControl\utils\data\FP_14k_24k.h5')

