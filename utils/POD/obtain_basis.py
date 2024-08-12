# PACKAGES
import numpy as np
from optht import optht
import random

# LOCAL FUNCTIONS
from utils.POD.fits import *
from utils.modelling.errors_flow import *


def prepare_POD_snapshot(D, N_t_POD):

    N_t = np.shape(D)[1]
    i_snps = random.sample([*range(N_t)], N_t_POD)

    return D[:, i_snps]

def get_ROM(D, r_method, r_threshold, dDdt=[]):
    """
    Retrieves POD complete and truncated basis

    :param D: snapshot matrix
    :param r_method: flag to select truncation method
    :param r_threshold: corresponding threshold value for given truncation method
    :param dDdt: derivative of snapshot matrix
    :return: POD dictionary
    """

    # PARAMETERS
    N_t = D.shape[1]

    # POD
    Phi, Sigma, Psi = np.linalg.svd(D, full_matrices=False)
    a = np.dot(np.diag(Sigma), Psi)

    # TRUNCATION METHOD
    if r_method == 'energy':
        n_r = energy_truncation(Sigma, r_threshold)
    elif r_method == 'elbow':
        E = np.cumsum(Sigma**2) / np.sum(Sigma**2)
        n_r = elbow_fit(np.arange(1, N_t+1), E)
    elif r_method == 'optimal':
        n_r = optht(D.shape[0]/N_t, Sigma, sigma=None)
    elif r_method == 'manual':
        n_r = r_threshold

    # TRUNCATE BASIS (note that nr is number of modes)
    Phir = Phi[:, 0:n_r]
    Sigmar = Sigma[0:n_r]
    Psir = Psi[0:n_r, :]
    ar = a[0:n_r, :]

    POD = {'Phir': Phir, 'Sigmar': Sigmar, 'Psir': Psir, 'ar': ar, 'Sigma': Sigma}

    # PROJECT ACCELERATION FIELDS ONTO POD BASIS (if needed)
    if dDdt:
        dPsir = np.dot(np.dot(np.linalg.inv(np.diag(Sigmar)), Phir.T), dDdt)
        dar = np.dot(Phir.T, dDdt)

        POD['dPsir'] = dPsir
        POD['dar'] = dar

    return POD

def get_rerr(Phi, Sigma, Psi, D):
    """
    Get reconstruction error array up to n_r modes
    :param Phi: spatial modes
    :param Sigma: singular value array
    :param Psi: temporal modes (transposed)
    :param D: snapshot matrix
    :return: error array
    """

    # PARAMETERS
    N_r = np.shape(Phi)[1]
    Sigma = np.diag(Sigma)

    err = np.zeros((N_r))

    # GET POD RECONSTRUCTION ERROR FOR EACH TRUNCATED BASIS
    for i in range(N_r):

        Dr = np.dot(Phi[:, 0:i+1], np.dot(Sigma[0:i+1, 0:i+1], Psi[0:i+1, :]))
        err[i] = np.sum(D**2 - Dr**2) / np.sum(D**2)

    return err


def get_cumenergy(Sigma):
    """
    Retrieves cumulative energy
    :param Sigma: singular value array
    :return: cumulative energy array
    """

    E = np.cumsum(Sigma ** 2) / np.sum(Sigma ** 2)

    return E