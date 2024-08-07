# PACKAGES
import numpy as np
import tensorflow as tf

# LOCAL FUNCTIONS
from utils.data.transformer import CNNAE2raw, raw2CNNAE


def get_modes_AE(AE, grid, D, N_z, flag_AE, flag_control, flag_static, z=0, b=0):
    """
    Retrives equivalent AE modes, depending on AE type
    :param AE: AE model class
    :param grid: dictionary containing X, Y grids with mask
    :param D: snapshot matrix (N_v, N_t)
    :param N_z: number of latent coordinates
    :param flag_AE: AE type
    :param flag_control: 1 if control is embedded in AE, 0 otherwise
    :param flag_static: 1 if modes are retrieved as static (latent space forced to 0s and 1s), 0 otherwise
    :param z: latent space, if needed
    :param b: control vector, if needed
    :return: AE modes (N_v, N_t, N_z)
    """

    # Parameters & initialization
    X_test = raw2CNNAE(grid, D)

    N_v = np.shape(X_test)[1] * np.shape(X_test)[2] * np.shape(X_test)[3]
    N_t = np.shape(X_test)[0]

    if flag_static:
        Phi = np.zeros((N_v, 1, N_z))
    else:
        Phi = np.zeros((N_v, N_t, N_z))

    # Retrieve each mode
    for i in range(N_z):

        if flag_AE=='MD-CNN-AE':

            if flag_static:
                X_mode = getattr(AE, 'decoder' + str(i + 1))(tf.ones([1, 1]))
            else:
                X_mode = AE.extract_mode(X_test, i + 1)

        elif flag_AE=='CNN-HAE':

            if flag_static:
                z = np.zeros((1, i + 1))
                z[:, i] = 1
                X_mode = AE['m' + str(i + 1)].decoder(z)
            else:
                if i == 0:
                    X_mode = AE['m' + str(i + 1)]([X_test])
                else:
                    X_mode = AE['m' + str(i + 1)]([X_test, np.zeros((N_t, i))])

        elif (flag_AE=='CNN-VAE') or (flag_AE=='C-CNN-AE'):

            if flag_static:
                aux_z = np.zeros((1, N_z))
                aux_z[:, i] = 1
            else:
                aux_z = np.zeros(np.shape(z))
                aux_z[:, i] = z[:, i]

            if flag_control:
                X_mode = AE.decoder(tf.keras.layers.Concatenate(axis=1)([aux_z, b]))
            else:
                X_mode = AE.decoder(aux_z)

        Phi[:, :, i] = CNNAE2raw(X_mode)

    return Phi


def get_correlation_matrix(Phi):
    """
    Obtains correlation matrix for AE modes
    :param Phi: AE modes (N_v, N_t, N_z)
    :return: determinant of correlation matrix and correlation matrx
    """

    N_z = np.shape(Phi)[2]
    N_t = np.shape(Phi)[1]

    Cij = np.zeros((N_z, N_z, N_t))
    Rij = np.zeros((N_z, N_z, N_t))
    detR = np.zeros((N_t))

    for t in range(N_t):
        PHI = Phi[:,t,:].T
        Cij[:, :, t] = np.abs(np.cov(PHI))

        for i in range(N_z):
            for j in range(N_z):
                Rij[i,j,t] = Cij[i,j,t] / np.sqrt(Cij[i,i,t] * Cij[j,j,t])

        detR[t] = np.linalg.det(Rij[:,:,t])


    return detR, Rij

