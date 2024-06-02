# PACKAGES
import numpy as np
import warnings
import matplotlib.pyplot as plt
import random
import time

# LOCAL FILES
from utils.data.read_data import read_FP
from utils.POD.fits import elbow_fit, energy_truncation
from utils.POD.obtain_basis import get_cumenergy

# PARAMETERS
path_flow_train = r'/utils/data/FPc_00k_80k.h5'
path_flow_test = r'/utils/data/FPc_00k_03k.h5'
path_grid = r'/utils/data/FP_grid.h5'
warnings.filterwarnings("ignore")

# LOAD DATA
grid, flow_train = read_FP(path_grid, path_flow_train)
flow_test = read_FP(path_grid, path_flow_test)[1]

# Get snapshot matrix
D_train = np.concatenate( (flow_train['U'], flow_train['V']), axis=0)
D_test = np.concatenate( (flow_test['U'], flow_test['V']), axis=0)

Dmean = np.mean(D_train, axis=1).reshape((np.shape(D_train)[0]),1)

Ddt_train = D_train - Dmean
Ddt_test = D_test - Dmean
del D_train, flow_train, D_test, flow_test

nt = np.shape(Ddt_train)[1]
n_modes = 1500

i_snps = random.sample([*range(nt)],n_modes)

# Get POD basis
t0 = time.time()
Phi_train, Sigma_train, Psi_train = np.linalg.svd(Ddt_train[:,i_snps], full_matrices=False)
t1 = time.time()

E = get_cumenergy(Sigma_train)

Psi_test = np.dot(np.linalg.inv(np.diag(Sigma_train)), np.dot(Phi_train.T, Ddt_test))
Ddt_test_r = np.dot(Phi_train, np.dot(np.diag(Sigma_train), Psi_test))

err_POD = np.mean((Ddt_test - Ddt_test_r)**2)

a = 0