# PACKAGES
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt

# LOCAL FILES
from utils.data.read_data import read_SC
from utils.data.transform_data import raw2CNNAE, CNNAE2raw
from utils.modelling.differentiation import diff_time
from utils.plt.plt_snps import *
from utils.POD.obtain_basis import get_ROM, get_rerr, get_cumenergy
from utils.AEs.AE_modes import get_modes_CNNHAE, get_modes_CNNHAE_static
from utils.AEs.AE_energy import energy_CNNHAE
from utils.AEs.AE_train import train_CNNHAE

# PARAMETERS
path_data = r'F:\AEs_wControl\utils\data'
path_rel = r'SC_00k_00k_AE.mat'
warnings.filterwarnings("ignore")

n_epochs = 100  # number of epoch
nstrides = 2
ksize = (3,3)
psize = (2,2)
ptypepool = 'valid'
nr = 6 # Number of nonlinear modes

# LOAD DATA
grid, flow = read_SC(os.path.join(path_data, path_rel))

# Get snapshot matrix
D = np.concatenate( (flow['U'], flow['V']), axis=0)
Dmean = np.mean(D, axis=1).reshape((np.shape(D)[0]),1)
Ddt = D - Dmean
dDdt = diff_time(D,flow['t'])

# Get POD basis
flag_truncate =['elbow',]
POD = get_ROM(grid, Ddt, dDdt, flag_truncate)

E = get_cumenergy(POD['Sigma'])
err_energy = get_rerr(POD['Phi'], POD['Sigma'], POD['Psi'], Ddt, grid['B'])

# Prepare data for autoencoder
X_train, X_test, y_train, y_test = raw2CNNAE(grid, Ddt, flag_split=1)

AE, lat_vector = train_CNNHAE(n_epochs, ksize, psize, ptypepool, nstrides, nr, X_train, X_test, X_test, y_train, y_test)

# Get modes from autoencoder
Phi1_AE, Phi2_AE = get_modes_CNNHAE(AE, X_test, lat_vector, nr)
Phi1_AE_static, Phi2_AE_static = get_modes_CNNHAE_static(AE, X_test, nr)

# Get truncated and true test set
D_test = CNNAE2raw(X_test)
a_test = np.dot(POD['Phi'].T, D_test)
Dr_test_POD = np.dot(POD['Phi'][:,0:nr], a_test[0:nr, :])

# Get energy estimations
cum_energy, energy = energy_CNNHAE(D_test, Phi1_AE, POD['Phi'], a_test)

plot_video_snp(grid, Ddt)