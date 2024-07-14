import numpy as np

def get_reyn_stresses_2D(grid, Ddt):

    X = grid['X']

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    U = Ddt[0: m * n, :]
    V = Ddt[n * m: n * m * 2, :]

    REYN = {}
    REYN['uu'] = np.mean(U ** 2, axis=1)
    REYN['uv'] = np.mean(U * V, axis=1)
    REYN['vv'] = np.mean(V ** 2, axis=1)
    REYN['TKE'] = 1 / 2 * (REYN['uu'] + REYN['vv'])

    return REYN

def get_energy_fluctuations(Ddt):

    nt = np.shape(Ddt)[1]
    E = np.zeros((nt))

    Phi, Sigma, Psi = np.linalg.svd(Ddt, full_matrices=False)

    for t in range(nt):
        E[t] = 1 / 2 * np.dot(Ddt[:, t], Ddt[:, t])

    EPOD = 1 / 2 * np.sum(Psi ** 2, axis=0)
    # Check that it´s the same first!

    return E