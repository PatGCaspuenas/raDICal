import numpy as np
from optht import optht

from utils.POD.fits import *
from utils.modelling.errors import *

def get_ROM(grid, D, dDdt, flag_truncate):

    nt = D.shape[1]

    Phi, Sigma, Psi = np.linalg.svd(D, full_matrices=False)
    a = np.dot(np.diag(Sigma), Psi)

    # Choose truncation method
    if flag_truncate[0] == 'energy':
        nr = energy_truncation(Sigma, flag_truncate[1])
    elif flag_truncate[0] == 'elbow':
        E = np.cumsum(Sigma**2) / np.sum(Sigma**2)
        nr = elbow_fit(np.arange(1, nt+1), E)
    elif flag_truncate[0] == 'optimal':
        nr = optht(D.shape[0]/nt, Sigma, sigma=None)
    elif flag_truncate[0] == 'manual':
        nr = flag_truncate[1]

    # Truncate basis
    Phir = Phi[:, 0:nr]
    Sigmar = Sigma[0:nr]
    Psir = Psi[0:nr, :]
    ar = a[0:nr, :]

    # Project acceleration fields onto POD basis
    dPsir = np.dot(np.dot(np.linalg.inv(np.diag(Sigmar)), Phir.T), dDdt)
    dar = np.dot(Phir.T, dDdt)

    POD = {'Phir': Phir, 'Sigmar': Sigmar, 'Psir': Psir, 'ar': ar, 'dPsir':dPsir, 'dar': dar, 'Phi': Phi, 'Sigma': Sigma, 'Psi': Psi, 'a': a}
    return POD

def get_rerr(Phi, Sigma, Psi, D, B):

    nr = np.shape(Phi)[1]
    Sigma = np.diag(Sigma)

    err = np.zeros((nr))

    for i in range(nr):

        Dr = np.dot(Phi[:, 0:i+1], np.dot(Sigma[0:i+1, 0:i+1], Psi[0:i+1, :]))
        err[i] = np.sum(D**2 - Dr**2) / np.sum(D**2)

    return err


def get_cumenergy(Sigma):

    E = np.cumsum(Sigma ** 2) / np.sum(Sigma ** 2)

    return E