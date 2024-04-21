# PACKAGES
import numpy as np
from sklearn.model_selection import train_test_split
import array

def raw2CNNAE(grid, D, flag_split=0):

    # Read and transform data
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

    U = np.zeros((nt, n, m, k))
    for i in range(k):
        U[:, :, :, i] = np.reshape(D[( (n * m)*i ):( (n * m)*(i + 1) ), :], (m, n, nt), order='F').T

    if flag_split:
        X_train, X_test, y_train, y_test = train_test_split(U, U, test_size=0.3, random_state=1, shuffle=False)
        return X_train, X_test, y_train, y_test
    else:
        return U, U

def CNNAE2raw(U):

    m = np.shape(U)[1]
    n = np.shape(U)[2]
    nt = np.shape(U)[0]
    k = np.shape(U)[3]

    u = np.array(U).T # (k, m, n, nt)
    D = np.zeros((k*m*n, nt))

    for i in range(k):

        D[( (n * m)*i ):( (n * m)*(i + 1) ), :] = np.reshape(u[i, :, :, :], (m*n, nt), order='F')

    return D