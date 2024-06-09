# PACKAGES
import numpy as np
from sklearn.model_selection import train_test_split

def raw2CNNAE(grid, D, flag_split=0, flag_control=0, u=0):

    # FLOW PARAMETERS
    X = grid['X']
    Y = grid['Y']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(D)[1]

    if np.shape(D)[0] == n * m:
        k = 1
    elif np.shape(D)[0] == 2 * n * m:
        k = 2
    else:
        k = 3

    # CONTROL PARAMETERS
    if flag_control:
        nc = np.shape(u)[1]

    # RESHAPE FLOW (control is already reshaped)
    U = np.zeros((nt, n, m, k))
    for i in range(k):
        U[:, :, :, i] = np.reshape(D[( (n * m)*i ):( (n * m)*(i + 1) ), :], (m, n, nt), order='F').T

    # SPLIT AND RETURN
    if flag_split and flag_control:
        X_train, X_test, u_train, u_test = train_test_split(U, u, test_size=0.3, shuffle=False)
        return X_train, X_test, u_train, u_test
    elif flag_split and ~flag_control:
        X_train, X_test = train_test_split(U, test_size=0.3, shuffle=False)
        return X_train, X_test, 0, 0
    else:
        return U

def CNNAE2raw(U):

    # PARAMETERS
    m = np.shape(U)[1]
    n = np.shape(U)[2]
    nt = np.shape(U)[0]
    k = np.shape(U)[3]

    # INITIALIZATION
    u = np.array(U).T # (k, m, n, nt)
    D = np.zeros((k*m*n, nt))

    # RESHAPE
    for i in range(k):

        D[( (n * m)*i ):( (n * m)*(i + 1) ), :] = np.reshape(u[i, :, :, :], (m*n, nt), order='F')

    return D

def get_control_vector(flow, flag_flow, flag_control):

    if flag_flow=='FP':
        nt = np.shape(flow['U'])[1]
        if flag_control:
            if 'vF' in flow:
                u = np.concatenate((flow['vF'], flow['vT'], flow['vB']), axis=1)
            else:
                u = np.zeros((nt, 3))
        else:
            u = np.zeros((nt, 3))
    else:
        u = 0

    return u

