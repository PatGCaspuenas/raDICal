# PACKAGES
import numpy as np
import warnings
import matplotlib.pyplot as plt
import random

# LOCAL FILES
from utils.data.read_data import read_FP
from utils.POD.fits import elbow_fit, energy_truncation
from utils.POD.obtain_basis import get_cumenergy

# PARAMETERS
path_flow = r'/utils/data/FPc_00k_80k.h5'
path_grid = r'/utils/data/FP_grid.h5'
warnings.filterwarnings("ignore")

# LOAD DATA
grid, flow = read_FP(path_grid, path_flow)

# Get snapshot matrix
D = np.concatenate( (flow['U'], flow['V']), axis=0)
Dmean = np.mean(D, axis=1).reshape((np.shape(D)[0]),1)
Ddt = D - Dmean
del D, flow

nt = np.shape(Ddt)[1]
n_modes = [50, 100, 500, 1000, 1500, 3000, 5000]
energies = [50, 60, 70, 80, 90, 95, 99]
energy_nr = {'e50': [], 'e60': [], 'e70': [], 'e80': [], 'e90': [], 'e95': [], 'e99': [], 'elbow': []}
elbow_energy = []
for i in n_modes:

    i_snps = random.sample([*range(nt)],i)

    # Get POD basis
    Phi, Sigma, Psi = np.linalg.svd(Ddt[:,i_snps], full_matrices=False)

    # Get energy values
    for j in energies:
        energy_nr['e' + str(j)].append(energy_truncation(Sigma, j/100))

    E = get_cumenergy(Sigma)
    energy_nr['elbow'].append(elbow_fit(np.arange(1, i + 1), E))
    elbow_energy.append(E[elbow_fit(np.arange(1, i + 1), E) - 1])


colors_plot = ['mediumblue', 'darkorchid', 'deeppink', 'coral', 'sienna', 'seagreen', 'teal', 'lightslategray']
fig, ax = plt.subplots(1, 1, subplot_kw=dict(box_aspect=1))

for j,k in zip(energies,colors_plot[:-1]):
    ax.loglog(n_modes, energy_nr['e' + str(j)], 'o-', color=k, label=str(j)+'%')
ax.loglog(n_modes, energy_nr['elbow'], 'o-', color=colors_plot[-1], label='elbow')

ax.set_xlabel('$n_t$')
ax.set_ylabel('$n_r$')

ax.axis([n_modes[0], n_modes[-1], 1, 100])

ax.legend()
plt.show()

a=0