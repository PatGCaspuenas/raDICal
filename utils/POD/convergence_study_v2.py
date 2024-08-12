# PACKAGES
import numpy as np
import warnings
import scipy.io as sio
import random
import pickle

# LOCAL FILES
from utils.data.read_data import read_FP
from utils.POD.fits import elbow_fit, energy_truncation
from utils.POD.obtain_basis import get_cumenergy

# PARAMETERS
paths_flow = [r'F:\AEs_wControl\DATA\FPc_00k_70k.h5', r'F:\AEs_wControl\DATA\FPc_00k_50k.h5', r'F:\AEs_wControl\DATA\FPc_00k_30k.h5',
              r'F:\AEs_wControl\DATA\FPc_00k_70k.h5']
paths_flow_test = [r'F:\AEs_wControl\DATA\FPc_00k_03k.h5', r'F:\AEs_wControl\DATA\FPc_00k_03k.h5', r'F:\AEs_wControl\DATA\FPc_00k_03k.h5',
              r'F:\AEs_wControl\DATA\FPc_00k_03k.h5']
paths_mean = [r'F:\AEs_wControl\DATA\others\FPc_Dmean.npy', r'F:\AEs_wControl\DATA\others\FPc_Dmean.npy', r'F:\AEs_wControl\DATA\others\FPc_Dmean.npy',
              r'F:\AEs_wControl\DATA\others\FPc_Dmean.npy']
paths_flow = [r'F:\AEs_wControl\DATA\FP_00k_27k.h5', r'F:\AEs_wControl\DATA\FP_00k_20k.h5',
              r'F:\AEs_wControl\DATA\FP_00k_10k.h5']
paths_flow_test = [r'F:\AEs_wControl\DATA\FP_27k_30k.h5', r'F:\AEs_wControl\DATA\FP_27k_30k.h5',
              r'F:\AEs_wControl\DATA\FP_27k_30k.h5']
paths_mean = [r'F:\AEs_wControl\DATA\others\FP_Dmean.npy', r'F:\AEs_wControl\DATA\others\FP_Dmean.npy',
              r'F:\AEs_wControl\DATA\others\FP_Dmean.npy']
path_grid = r'F:\AEs_wControl\DATA\FP_grid.h5'
warnings.filterwarnings("ignore")

# LOAD DATA
FLOW = {}
j = 0
for path_flow, path_test, path_mean in zip(paths_flow, paths_flow_test, paths_mean):
    print(j)

    FLOW[str(j)] = {}

    grid, flow = read_FP(path_grid, path_flow)
    flow_test = read_FP(path_grid, path_test)[1]

    # Get snapshot matrix
    D_test = np.concatenate((flow_test['U'], flow_test['V']), axis=0)
    D = np.concatenate( (flow['U'], flow['V']), axis=0)
    Dmean = np.load(path_mean).reshape(-1,1)
    Ddt = D - Dmean
    Ddt_test = D_test - Dmean
    del D, flow, D_test, flow_test

    nt = np.shape(Ddt)[1]
    n_modes = [50, 100, 500, 1000, 1500, 3000, 5000]
    energies = [50, 60, 70, 80, 90, 95, 99]
    energy_nr = {'e50': [], 'e60': [], 'e70': [], 'e80': [], 'e90': [], 'e95': [], 'e99': [], 'elbow': []}
    elbow_energy = {}
    err_gen = {}
    Sigmas = {}
    for i in n_modes:
        print(i)

        i_snps = random.sample([*range(nt)],i)

        # Get POD basis
        Phi, Sigma, Psi = np.linalg.svd(Ddt[:,i_snps], full_matrices=False)

        # Get energy values
        for k in energies:
            energy_nr['e' + str(k)].append(energy_truncation(Sigma, k/100))

        E = get_cumenergy(Sigma)
        energy_nr['elbow'].append(elbow_fit(np.arange(1, i + 1), E))
        elbow_energy[str(i)] = (E[elbow_fit(np.arange(1, i + 1), E) - 1])

        Psi_test = np.dot(np.linalg.inv(np.diag(Sigma)), np.dot(Phi.T, Ddt_test))
        Ddt_test_r = np.dot(Phi, np.dot(np.diag(Sigma), Psi_test))

        err_gen[str(i)] = (np.mean((Ddt_test - Ddt_test_r) ** 2))
        Sigmas[str(i)] = Sigma

    FLOW[str(j)]['elbow_energy'] = elbow_energy
    FLOW[str(j)]['err_gen'] = err_gen
    FLOW[str(j)]['energy_nr'] = energy_nr
    FLOW[str(j)]['Sigmas'] = Sigmas

    j += 1

with open('POD_convergence_FP', "wb") as fp:  # Pickling
    pickle.dump(FLOW, fp)
a=0