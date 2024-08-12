import numpy as np
import tensorflow as tf
import random
from scipy.signal import savgol_filter

from utils.data.transform_data import raw2CNNAE, CNNAE2raw


def get_AE_z(nr, flag_AE, AE, grid, D_test):

    X_test = raw2CNNAE(grid, D_test)
    nt = np.shape(X_test)[0]
    nt_max = 500
    mult = (np.floor(nt / nt_max)).astype(int)

    if nt > nt_max:
        ii = [0]
        for i in range(mult):
            ii.append(ii[-1] + nt_max)
        ii.append(nt-1)
    else:
        ii = [0, nt-1]

    if flag_AE == 'CNN-HAE':

        for i in range(nr):

            if i == 0:
                z_test = AE['m' + str(i + 1)].get_latent_vector(X_test)
            else:
                z_test = tf.keras.layers.Concatenate(axis=1)([z_test, AE['m' + str(i + 1)].get_latent_vector(X_test)])

    else:

        z_test = np.zeros((nt, nr))
        for i in range(len(ii) - 1):
            z_test[ii[i]:ii[i + 1], :] = AE.get_latent_vector(X_test[ii[i]:ii[i + 1], :, :, :])
        z_test = tf.convert_to_tensor(z_test)

    return z_test

def filter_AE_z(z_test):

    nr = np.shape(z_test)[1]
    for i in range(nr):
        z_test[:, i] = savgol_filter(z_test[:,i], 15, 3)

    return z_test

def get_AE_reconstruction(nr, flag_AE, flag_control, AE, grid, D_test, b_test, flag_filter = 0):

    X_test = raw2CNNAE(grid, D_test)
    nt = np.shape(X_test)[0]
    nt_max = 500
    mult = (np.floor(nt / nt_max)).astype(int)

    if nt > nt_max:
        ii = [0]
        for i in range(mult):
            ii.append(ii[-1] + nt_max)
        ii.append(nt)
    else:
        ii = [0, nt]

    if flag_AE == 'CNN-HAE':

        z_test = AE['m' + str(nr-1)].get_latent_vector(X_test)

        y_test = AE['m' + str(nr)]([X_test, tf.convert_to_tensor(z_test.numpy())])

    elif flag_AE == 'MD-CNN-AE':

        y_test = AE(X_test)

    else:
        z_test = np.zeros((nt,nr))
        for i in range(len(ii)-1):
            z_test[ii[i]:ii[i+1], :] = AE.get_latent_vector(X_test[ii[i]:ii[i+1], :, :, :])

        if flag_filter:
            z_test = filter_AE_z(z_test)
            z_test = tf.convert_to_tensor(z_test)

        y_test = np.zeros_like(X_test)
        if flag_control:
            for i in range(len(ii) - 1):
                y_test[ii[i]:ii[i+1], :, :, :] = AE.decoder((tf.keras.layers.Concatenate(axis=1)([z_test[ii[i]:ii[i+1], :], tf.convert_to_tensor(b_test[ii[i]:ii[i+1], :])])))
        else:
            for i in range(len(ii) - 1):
                y_test[ii[i]:ii[i+1], :, :, :] = AE.decoder(z_test[ii[i]:ii[i+1], :])

    Dr_test = CNNAE2raw(y_test)

    return Dr_test
