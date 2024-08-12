# PACKAGES
import numpy as np

def get_reyn_stresses_2D(grid, Ddt):
    """
    Retrieves Reynold stresses for 2D flow

    :param grid: dictionary containing X, Y grids
    :param Ddt: snapshot matrix of fluctuations of velocity
    :return: dictionary containing TKE and Reynolds stresses
    """

    X = grid['X']

    N_y, N_x = np.shape(X)

    U = Ddt[0: N_x * N_y, :]
    V = Ddt[N_x * N_y:N_x * N_y * 2, :]

    REYN = {}
    REYN['uu'] = np.mean(U ** 2, axis=1)
    REYN['uv'] = np.mean(U * V, axis=1)
    REYN['vv'] = np.mean(V ** 2, axis=1)
    REYN['TKE'] = 1 / 2 * (REYN['uu'] + REYN['vv'])

    return REYN

def get_energy_fluctuations(D):
    """
    Retrieves energy fluctuations in time

    :param D: snapshot matrix of velocity (or fluctuations)
    :return: energy in time
    """

    N_t = np.shape(D)[1]
    E = np.zeros((N_t))

    for t in range(N_t):
        E[t] = 1 / 2 * np.dot(D[:, t], D[:, t])

    # Should be equivalent to the following snippet
    # Phi, Sigma, Psi = np.linalg.svd(D, full_matrices=False)
    # EPOD = 1 / 2 * np.sum(Psi ** 2, axis=0)

    return E