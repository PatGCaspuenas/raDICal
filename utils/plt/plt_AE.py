import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def plot_train_val_loss(model):

    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plot_energy(energy_POD, energy_AE, i_AE_energy):

    nr = len(energy_POD)

    fig, ax = plt.subplots(1,1, subplot_kw=dict(box_aspect=1))
    ax.loglog(np.arange(1,nr+1),energy_POD, 'bo-', label='POD')
    ax.loglog(np.arange(1,nr+1),energy_AE[i_AE_energy], 'mo-', label='MD-CNN-AE')
    ax.set_xlabel('$r$')
    ax.set_ylabel('EA')

    ax.axis([1, nr, np.min(np.concatenate((energy_POD, energy_AE))), 1])
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = FuncFormatter(lambda y, _: '{:.2g}'.format(y))
        axis.set_minor_formatter(formatter)
        axis.set_major_formatter(formatter)
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(1,1, subplot_kw=dict(box_aspect=1))
    ax.loglog(np.arange(1,nr+1),np.cumsum(energy_POD), 'bo-', label='POD')
    ax.loglog(np.arange(1,nr+1),np.cumsum(energy_AE[i_AE_energy]), 'mo-', label='MD-CNN-AE')
    ax.set_xlabel('$r$')
    ax.set_ylabel('CEA')

    ax.axis([1, nr, np.min([np.cumsum(energy_POD)[0], np.cumsum(energy_AE[i_AE_energy])[0]]), 1])
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = FuncFormatter(lambda y, _: '{:.2g}'.format(y))
        axis.set_minor_formatter(formatter)
        axis.set_major_formatter(formatter)
    ax.legend()
    plt.show()

def plot_corr_matrix(Rij, detR):

    nr = np.shape(Rij)[0]

    fig, ax = plt.subplots(1, 1)

    cp0 = ax.imshow(Rij, cmap='magma_r', vmin=0, vmax=1)

    ax.set_xticks(np.arange(nr))
    ax.set_yticks(np.arange(nr))
    ax.set_xticklabels(np.arange(1, nr + 1))
    ax.set_yticklabels(np.arange(1, nr + 1))

    ax.set_title('$detR = ' + '{0:.2f}'.format((detR * 100)) + ' \% $')
    fig.colorbar(cp0, ticks=[0, 0.5, 1])

    plt.show()