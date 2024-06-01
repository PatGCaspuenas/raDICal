import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter

from utils.POD.obtain_basis import get_cumenergy

def plot_phi(grid, Phi):

    X = grid['X']
    Y = grid['Y']
    B = grid['B']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nr = np.shape(Phi)[1]

    if np.shape(Phi)[0] == n * m:
        k = 1
    elif np.shape(Phi)[0] == 2 * n * m:
        k = 2
    else:
        k = 3

    M = np.zeros((m, n, nr, k))
    for i in range(k):
        M[:, :, :, i] = np.reshape(Phi[( (n * m)*i ):( (n * m)*(i + 1) ), :], (m, n, nr), order='F')

    cticks = np.linspace(-0.5, 0.5, 3)
    clevels = [-0.5, 0.5]

    fig, ax = plt.subplots(k, nr, layout = 'tight')

    plt.tight_layout()
    for i in range(nr):
        for j in range(k):
            cp0 = ax[j, i].pcolormesh(X, Y, M[:, :, i, j].reshape(m,n), cmap = 'jet', vmin = clevels[0], vmax = clevels[1])
            cp00 = ax[j, i].contourf(X, Y, B, colors='k')

            ax[j, i].set_title('$\phi_{' + str(i) + '}$')
            ax[j, i].axis('scaled')
            ax[j, i].set_xlim([np.min(X), np.max(X)])
            ax[j, i].set_ylim([np.min(Y), np.max(Y)])

            if i == (nr-1):
                divider = make_axes_locatable(ax[j, i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar =fig.colorbar(cp0, ax=ax[j, i], ticks=cticks, cax=cax)

            if i == 0:
                ax[j, i].set_ylabel('$y/D$')
            else:
                ax[j, i].set_yticks([])

            if j == (k-1):
                ax[j, i].set_xlabel('$x/D$')
            else:
                ax[j, i].set_xticks([])

    plt.show()
    plt.tight_layout()

def plot_cum_energy_POD(Sigma):

    E = get_cumenergy(Sigma)

    nr = len(E)

    fig, ax = plt.subplots(1,1, subplot_kw=dict(box_aspect=1))
    ax.loglog(np.arange(1,nr+1),E, 'bo-', label='POD')
    ax.set_xlabel('$r$')
    ax.set_ylabel('CEA')

    ax.axis([1, nr, E[0], 1])

    ax.legend()
    plt.show()