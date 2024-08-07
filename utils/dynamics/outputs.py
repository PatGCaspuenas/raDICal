# PACKAGES
import numpy as np
import tensorflow as tf

# LOCAL FUNCTIONS
from utils.data.transformer import raw2dyn, CNNAE2raw
from utils.data.loader import load_model_AE

def get_predicted_z(PARAMS, FLAGS, DYN, z, t, b=0):
    """
    Predicts latent space
    :param PARAMS: dictionary of parameters
    :param FLAGS: dictionary of flags
    :param DYN: dynamical predictor model
    :param z: latent space in raw format (N_t, N_z)
    :param t: time vector (N_t,1)
    :param u: control vector, if needed (N_t, N_c)
    :return: TDL and PW latent and control space
    """

    flag_control_dyn = FLAGS['DYN']['control']
    flag_control = FLAGS['FLOW']['control']
    w_prop = PARAMS['DYN']['w_prop']

    # Prepare shape of latent and control space
    if not flag_control:
        zx_test, zy_test, T = raw2dyn(t, z, PARAMS, flag_control, flag_train=0)
    else:
        zx_test, zy_test, bx_test, by_test, T = raw2dyn(t, z, PARAMS, flag_control, flag_train=0, b=b)

    # Predict latent space
    if flag_control_dyn:
        zy_test_dyn = DYN.predict([zx_test, bx_test, by_test], w_prop)
    else:
        zy_test_dyn = DYN.predict(zx_test, w_prop)

    # Return corresponding variables
    if not flag_control:
        return zx_test, zy_test, zy_test_dyn, 0, 0, T
    else:
        return zx_test, zy_test, zy_test_dyn, bx_test, by_test, T



def get_predicted_flow(PARAMS, FLAGS, PATHS, zy, zy_dyn, by=0):
    """
    Decodes predicted latent space
    :param PARAMS: dictionary of parameters
    :param FLAGS: dictionary of flags
    :param PATHS: dictionary of paths
    :param zy: ground truth of PW latent space (N_w, w_prop, N_z)
    :param zy_dyn: predicted latent space (N_w, w_prop, N_z)
    :param by: control vector in PW
    :return: decoded flow with ground truth and predicted latent space (N_w, N_v, w_prop)
    """

    # PARAMS
    N_w = np.shape(zy)[0]
    N_t = np.shape(zy)[1]

    # FLAGS
    flag_control = FLAGS['AE']['control']

    # LOAD MODEL
    AE = load_model_AE(PARAMS, FLAGS, PATHS)

    # PREPARE FLOW
    N_y, N_x, K = AE.N_y, AE.N_x, AE.K

    D, Dr = np.zeros((2, N_w, N_x * N_y * K, N_t))

    # DECODE GROUND TRUTH & PREDICTED
    for w in range(N_w):
        if flag_control:
            y_test = AE.decoder((tf.keras.layers.Concatenate(axis=1)([zy[w, :, :], tf.convert_to_tensor(by[w, :, :])])))
            yr_test = AE.decoder((tf.keras.layers.Concatenate(axis=1)([zy_dyn[w, :, :], tf.convert_to_tensor(by[w, :, :])])))
        else:
            y_test = AE.decoder(zy[w, :, :])
            yr_test = AE.decoder(zy_dyn[w, :, :])

        D[w, :, :] = CNNAE2raw(y_test)
        Dr[w, :, :] = CNNAE2raw(yr_test)

    return D, Dr


