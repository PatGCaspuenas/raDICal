import numpy as np
import tensorflow as tf
import random

from utils.data.transform_data import raw2CNNAE, CNNAE2raw


def get_AE_z(nr, flag_AE, AE, grid, D_test):

    X_test = raw2CNNAE(grid, D_test)

    if flag_AE == 'CNN-HAE':

        for i in range(nr):

            if i == 0:
                z_test = AE['m' + str(i + 1)].get_latent_vector(X_test)
            else:
                z_test = tf.keras.layers.Concatenate(axis=1)([z_test, AE['m' + str(i + 1)].get_latent_vector(X_test)])

    else:

        z_test = AE.get_latent_vector(X_test)

    return z_test

def get_AE_reconstruction(nr, flag_AE, flag_control, AE, grid, D_test, b_test):

    X_test = raw2CNNAE(grid, D_test)

    if flag_AE == 'CNN-HAE':

        z_test = AE['m' + str(nr-1)].get_latent_vector(X_test)

        y_test = AE['m' + str(nr)]([X_test, tf.convert_to_tensor(z_test.numpy())])

    else:

        if flag_control:
            y_test = AE([X_test, tf.convert_to_tensor(b_test)])
        else:
            y_test = AE(X_test)

    Dr_test = CNNAE2raw(y_test)

    return Dr_test
