# PACKAGES
import numpy as np
import warnings
import matplotlib.pyplot as plt
import random

# LOCAL FILES
from utils.data.read_data import read_FP
from utils.modelling.differentiation import get_2Dvorticity
from utils.data.transform_data import raw2CNNAE, CNNAE2raw
from utils.modelling.differentiation import diff_time
from utils.plt.plt_snps import *
from utils.POD.obtain_basis import get_ROM, get_rerr, get_cumenergy
from utils.AEs.AE_modes import get_modes_CNNVAE, get_modes_CNNVAE_static
from utils.AEs.AE_energy import energy_CNNVAE
from utils.AEs.AE_train import train_CNNVAE

# PARAMETERS
Re = 130
path_flow = r'F:\AEs_wControl\utils\data\FP_14k_24k.h5'
path_flow_test = r'F:\AEs_wControl\utils\data\FP_10k_13k.h5'
path_grid = r'F:\AEs_wControl\utils\data\FP_grid.h5'
warnings.filterwarnings("ignore")

n_epochs = 100  # number of epoch
nstrides = 2
ksize = (3,3)
psize = (2,2)
ptypepool = 'valid'
nr = 5 # Number of nonlinear modes
beta = 1e-3
act = 'tanh'

# LOAD DATA
grid, flow = read_FP(path_grid, path_flow)
flow_test = read_FP(path_grid, path_flow_test)[1]

# Get snapshot matrix
D = np.concatenate( (flow['U'], flow['V']), axis=0)
Dmean = np.mean(D, axis=1).reshape((np.shape(D)[0]),1)
Ddt = D - Dmean
t = flow['t']
del D, flow

D_test = np.concatenate( (flow_test['U'], flow_test['V']), axis=0)
Ddt_test = D_test[:, 0:1000] - Dmean
t_test = flow_test['t'][0:1000]
del D_test, flow_test

# Get POD basis
nt = np.shape(Ddt)[1]
i_snps = random.sample([*range(nt)],500)

Phi, Sigma, Psi = np.linalg.svd(Ddt[:,i_snps], full_matrices=False)
E = get_cumenergy(Sigma)

# Prepare data for autoencoder
X_train, X_val = raw2CNNAE(grid, Ddt, flag_split=1)[0:2]
X_test = raw2CNNAE(grid, Ddt_test, flag_split=0)[0]

AE, lat_vector = train_CNNVAE(beta, n_epochs, ksize, psize, ptypepool, nstrides, act, nr, X_train, X_val, X_test)
del X_train, X_val, Ddt

# Get modes from autoencoder
Phi_AE = get_modes_CNNVAE(AE, X_test, lat_vector, nr)
Phi_AE_static = get_modes_CNNVAE_static(AE, X_test, nr)
del X_test

# Get truncated and true test set
a_test = np.dot(Phi.T, Ddt_test)
Dr_test_POD = np.dot(Phi[:,0:nr], a_test[0:nr, :])

# Get energy estimations
cum_energy, energy, i_energy_AE = energy_CNNVAE(AE, Ddt_test, lat_vector, Phi[:,0:nr], a_test[0:nr, :])
plot_snp(grid, Phi, limits = [-0.5, 0.5], make_axis_visible = [1, 1], show_title = 1, show_colorbar = 1, flag_type = 'FP')