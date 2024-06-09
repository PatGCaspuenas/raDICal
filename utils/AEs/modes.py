import numpy as np
import tensorflow as tf

from utils.data.transform_data import CNNAE2raw, raw2CNNAE


def get_modes_AE(AE, grid, D_test, nr, flag_AE, flag_control, flag_static, z_test=0, b_test=0):

    X_test = raw2CNNAE(grid, D_test)

    nv = np.shape(X_test)[1] * np.shape(X_test)[2] * np.shape(X_test)[3]
    nt = np.shape(X_test)[0]

    if flag_static:
        Phi = np.zeros((nv, 1, nr))
    else:
        Phi = np.zeros((nv, nt, nr))

    for i in range(nr):

        if flag_AE=='MD-CNN-AE':

            if flag_static:
                X_mode = getattr(AE, 'decoder' + str(i + 1))(tf.ones([1, 1]))
            else:
                X_mode = AE.extract_mode(X_test, i + 1)

        elif flag_AE=='CNN-HAE':

            if flag_static:
                z_test = np.zeros((1, i + 1))
                z_test[:, i] = 1
                X_mode = AE['m' + str(i + 1)].decoder(z_test)
            else:
                if i == 0:
                    X_mode = AE['m' + str(i + 1)]([X_test])
                else:
                    X_mode = AE['m' + str(i + 1)]([X_test, np.zeros((nt, i))])

        elif (flag_AE=='CNN-VAE') or (flag_AE=='C-CNN-AE'):

            if flag_static:
                aux_z = np.zeros((1, nr))
                aux_z[:, i] = 1
            else:
                aux_z = np.zeros(np.shape(z_test))
                aux_z[:, i] = z_test[:, i]

            if flag_control:
                X_mode = AE.decoder(tf.keras.layers.Concatenate(axis=1)([aux_z, b_test]))
            else:
                X_mode = AE.decoder(aux_z)

        Phi[:, :, i] = CNNAE2raw(X_mode)

    return Phi


def get_correlation_matrix(Phi):

    nr = np.shape(Phi)[2]
    nt = np.shape(Phi)[1]

    Cij = np.zeros((nr, nr, nt))
    Rij = np.zeros((nr, nr, nt))
    detR = np.zeros((nt))

    for t in range(nt):
        PHI = Phi[:,t,:].T
        Cij[:, :, t] = np.abs(np.cov(PHI))

        for i in range(nr):
            for j in range(nr):
                Rij[i,j,t] = Cij[i,j,t] / np.sqrt(Cij[i,i,t] * Cij[j,j,t])

        detR[t] = np.linalg.det(Rij[:,:,t])


    return detR, Rij

