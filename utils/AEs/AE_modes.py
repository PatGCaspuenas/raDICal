import numpy as np
import tensorflow as tf

from utils.data.transform_data import CNNAE2raw

def get_modes_MDCNNAE(AE, X_test, nr):

    D_test = CNNAE2raw(X_test)

    nv = np.shape(D_test)[0]
    nt = np.shape(D_test)[1]

    Phi = np.zeros((nv, nt, nr))

    for i in range(nr):

        X_mode = AE.extract_mode(X_test, i+1)
        Phi[:, :, i] = CNNAE2raw(X_mode)
        Dr = np.sum(Phi, axis=2)

    return Phi

def get_modes_MDCNNAE_static(AE, X_test, nr):

    D_test = CNNAE2raw(X_test)

    nv = np.shape(D_test)[0]

    Phi = np.zeros((nv, 1, nr))

    for i in range(nr):
        X_mode = getattr(AE, 'decoder'+str(i+1))(tf.ones([1, 1]))
        Phi[:, :, i] = CNNAE2raw(X_mode)
        Dr = np.sum(Phi, axis=2)

    return Phi
def get_modes_CNNHAE(AE, X_test, lat_vector_test, nr):

    D_test = CNNAE2raw(X_test)

    nv = np.shape(D_test)[0]
    nt = np.shape(D_test)[1]

    Phi1 = np.zeros((nv, nt, nr))
    Phi2 = np.zeros((nv, nt, nr))

    for i in range(nr):
        if i == 0:
            Xr = AE['m' + str(i + 1)].get_reconstruction([X_test])
            Xmode = AE['m' + str(i + 1)].get_reconstruction([X_test])
        else:
            Xr = AE['m' + str(i + 1)].get_reconstruction([X_test, lat_vector_test[:, 0:i]])
            Xmode = AE['m' + str(i + 1)].get_reconstruction([X_test, np.zeros((nt,i))])

        Dr = CNNAE2raw(Xr)

        Phi1[:, :, i] = Dr - np.sum(Phi1, axis=2)
        Phi2[:, :, i] = CNNAE2raw(Xmode)

    return Phi1, Phi2

def get_modes_CNNHAE_static(AE, X_test, nr):

    D_test = CNNAE2raw(X_test)

    nv = np.shape(D_test)[0]

    Phi1 = np.zeros((nv, 1, nr))
    Phi2 = np.zeros((nv, 1, nr))

    for i in range(nr):
        lat_vector = np.zeros((1, i+1))
        lat_vector[:, i] = 1

        Xr = AE['m' + str(i + 1)].decoder(np.ones((1, i+1)))
        Xmode = AE['m' + str(i + 1)].decoder(lat_vector)

        Dr = CNNAE2raw(Xr)

        Phi1[:, :, i] = Dr - np.sum(Phi1, axis=2)
        Phi2[:, :, i] = CNNAE2raw(Xmode)

    return Phi1, Phi2

def get_modes_CNNVAE(AE, X_test, lat_vector_test, nr):

    D_test = CNNAE2raw(X_test)

    nv = np.shape(D_test)[0]
    nt = np.shape(D_test)[1]

    Phi = np.zeros((nv, nt, nr))

    for i in range(nr):
        aux_lat_vector = np.zeros(np.shape(lat_vector_test))
        aux_lat_vector[:, i] = lat_vector_test[:, i]

        X_mode = AE.decoder(aux_lat_vector)
        Phi[:, :, i] = CNNAE2raw(X_mode)

    return Phi


def get_modes_CNNVAE_static(AE, X_test, nr):
    D_test = CNNAE2raw(X_test)

    nv = np.shape(D_test)[0]

    Phi = np.zeros((nv, 1, nr))

    for i in range(nr):
        aux_lat_vector = np.zeros((1,nr))
        aux_lat_vector[:, i] = 1

        X_mode = AE.decoder(aux_lat_vector)
        Phi[:, :, i] = CNNAE2raw(X_mode)

    return Phi

def get_correlation_matrix(Phi):

    nr = np.shape(Phi)[2]

    PHI = Phi[:,0,:].T
    Cij = np.abs(np.cov(PHI))

    Rij = np.zeros((nr,nr))
    for i in range(nr):
        for j in range(nr):
            Rij[i,j] = Cij[i,j] / np.sqrt(Cij[i,i] * Cij[j,j])

    detR = np.linalg.det(Rij)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    cp0 = ax.imshow(Rij, cmap='magma_r', vmin=0, vmax=1)
    ax.set_xticks(np.arange(nr))
    ax.set_yticks(np.arange(nr))
    ax.set_xticklabels(np.arange(1, nr + 1))
    ax.set_yticklabels(np.arange(1, nr + 1))
    ax.set_title('$detR = ' + '{0:.2f}'.format((detR * 100)) + ' \% $')
    fig.colorbar(cp0, ticks=[0, 0.5, 1])
    plt.show()

    return detR, Rij

