# PACKAGES
import numpy as np
import tensorflow as tf

# LOCAL FUNCTIONS
from utils.data.transformer import raw2CNNAE, CNNAE2raw

def get_AE_z(N_z, flag_AE, AE, grid, D):
    """
    Gets latent space from AE

    :param N_z: number of latent coordinates
    :param flag_AE: AE type
    :param AE: AE class model
    :param grid: dictionary containing X,Y and mask grids
    :param D: snapshot matrix
    :return: latent space array
    """

    # PARAMETERS & INITIALIZATION
    X_test = raw2CNNAE(grid, D)
    N_t = np.shape(X_test)[0]
    N_t_max = 1000
    mult = (np.floor(N_t / N_t_max)).astype(int)

    # Indices for slicing data every 1000 snapshots (memory issues)
    if N_t > N_t_max:
        ii = [0]
        for i in range(mult):
            ii.append(ii[-1] + N_t_max)
        ii.append(N_t-1)
    else:
        ii = [0, N_t-1]

    if flag_AE == 'CNN-HAE':

        z = AE['m' + str(N_z-1)].get_latent_vector(X_test)

    else:

        z = np.zeros((N_t, N_z))
        for i in range(len(ii) - 1):
            z[ii[i]:ii[i + 1], :] = AE.get_latent_vector(X_test[ii[i]:ii[i + 1], :, :, :])
        z = tf.convert_to_tensor(z)

    return np.array(z)

def get_AE_reconstruction(N_z, flag_AE, flag_control, AE, grid, D, b):
    """
    Reconstructs flow field through AE (complete encoding-decoding process)

    :param N_z: number of latent coordinates
    :param flag_AE: AE type
    :param flag_control: 1 if control is embedded in AE, 0 otherwise
    :param AE: AE class model
    :param grid: dictionary containing X,Y and mask grids
    :param D: snapshot matrix
    :param b: control vector, if needed
    :return: reconstructed snapshot matrix
    """

    # PARAMETERS & INITIALIZATION
    X_test = raw2CNNAE(grid, D)
    N_t = np.shape(X_test)[0]
    N_t_max = 1000
    mult = (np.floor(N_t / N_t_max)).astype(int)

    # Indices for slicing data every 1000 snapshots (memory issues)
    if N_t > N_t_max:
        ii = [0]
        for i in range(mult):
            ii.append(ii[-1] + N_t_max)
        ii.append(N_t)
    else:
        ii = [0, N_t]

    # Encoding
    z = get_AE_z(N_z, flag_AE, AE, grid, D)

    # Decoding
    if flag_AE == 'CNN-HAE':

        y = AE['m' + str(N_z)]([X_test, tf.convert_to_tensor(z)])

    elif flag_AE == 'MD-CNN-AE':

        y = AE(X_test)

    else:

        y = np.zeros_like(X_test)
        if flag_control:
            for i in range(len(ii) - 1):
                y[ii[i]:ii[i+1], :, :, :] = AE.decoder((tf.keras.layers.Concatenate(axis=1)([z[ii[i]:ii[i+1], :], tf.convert_to_tensor(b[ii[i]:ii[i+1], :])])))
        else:
            for i in range(len(ii) - 1):
                y[ii[i]:ii[i+1], :, :, :] = AE.decoder(z[ii[i]:ii[i+1], :])

    Dr = CNNAE2raw(y)

    return Dr
