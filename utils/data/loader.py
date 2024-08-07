# PACKAGES
import tensorflow as tf
import numpy as np
import pickle
import os

# LOCAL FUNCTIONS
from utils.AEs.classes import CNN_VAE, C_CNN_AE
from utils.dynamics.classes import NARX, LSTM
from utils.modelling.custom_losses import energy_loss, null_loss, MeanSquaredError

def load_model_AE(PARAMS, FLAGS, PATHS):
    """
    Loads weigths for AE model
    :param PARAMS: dictionary of parameters
    :param FLAGS: dictionary of flags
    :param PATHS: dictionary of paths
    :return: AE class model
    """

    # DEFAULT PARAMS
    N_epochs = 1
    N_batch = 32
    l_r = 1e-3
    N_c = PARAMS['FLOW']['N_c']

    # FLAGS
    flag_AE = FLAGS['AE']['type']
    flag_control = FLAGS['AE']['control']

    # PATHS
    path_AE = PATHS['MODEL_AE']

    # DEFINE AE TYPE & LOAD WEIGHTS
    if flag_AE == 'CNN-VAE':
        AE = CNN_VAE(PARAMS, FLAGS)
    elif flag_AE == 'C-CNN-AE':
        AE = C_CNN_AE(PARAMS, FLAGS)

    # LOSS
    opt = tf.keras.optimizers.Adam(learning_rate=l_r)
    if (flag_AE=='CNN-VAE'):
        AE.compile(optimizer=opt, loss=null_loss, metrics=[energy_loss])
    else:
        AE.compile(optimizer=opt, loss='mse', metrics=[energy_loss])

    # INPUT
    if flag_control:
        input_train = [np.ones((N_batch, AE.N_x, AE.N_y, AE.K)), tf.convert_to_tensor(np.zeros((N_batch, N_c)))]
        input_val =   [np.ones((N_batch, AE.N_x, AE.N_y, AE.K)), tf.convert_to_tensor(np.zeros((N_batch, N_c)))]
    else:
        input_train = np.ones((N_batch, AE.N_x, AE.N_y, AE.K))
        input_val =   np.ones((N_batch, AE.N_x, AE.N_y, AE.K))

    # FIT
    AE.fit(input_train, np.ones((N_batch, AE.N_x, AE.N_y, AE.K)),
                                     epochs=N_epochs,
                                     shuffle=False,
                                     validation_data=(input_val, np.ones((N_batch, AE.N_x, AE.N_y, AE.K))),
                                     batch_size=N_batch)

    # LOAD WEIGTHS
    with open(path_AE[0], "rb") as fp:  # Unpickling
        enc = pickle.load(fp)
    with open(path_AE[1], "rb") as fp:  # Unpickling
        dec = pickle.load(fp)

    AE.encoder.set_weights(enc)
    AE.decoder.set_weights(dec)

    return AE

def load_model_dyn(PARAMS, FLAGS, PATHS):
    """
    Loads weigths for DYN model
    :param PARAMS: dictionary of parameters
    :param FLAGS: dictionary of flags
    :param PATHS: dictionary of paths
    :return: dynamical predictor class model
    """

    # PARAMETERS
    l_r = 1e-3
    N_epochs = 1
    N_batch = 32
    w_d = PARAMS['DYN']['w_d']
    w_prop = PARAMS['DYN']['w_prop']
    N_z = PARAMS['FLOW']['N_z']
    N_c = PARAMS['FLOW']['N_c']

    # FLAGS
    flag_control = FLAGS['DYN']['control']
    flag_type = FLAGS['DYN']['type']
    flag_opt = FLAGS['DYN']['optimizer']
    flag_loss = FLAGS['DYN']['loss']

    # PATHS
    path_dyn = PATHS['MODEL_DYN']

    # GENERATE WINDOW PREDICTIONS
    zx_train, zx_val = np.ones((2, N_batch, w_d, N_z))
    zy_train, zy_val = np.ones((2, N_batch, w_prop, N_z))
    if flag_control:
        bx_train, bx_val = np.ones((2, N_batch, w_d, N_c))
        by_train, by_val = np.ones((2, N_batch, w_prop, N_c))

    # CREATE DYNAMIC MODEL
    if flag_type == 'NARX':
        DYN = NARX(PARAMS, FLAGS)
    else:
        DYN = LSTM(PARAMS, FLAGS)

    # COMPILE
    if flag_opt == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=l_r)
        if flag_loss == 'mse':
            DYN.compile(optimizer=opt, loss='mse', metrics = ['mae'])
        else:
            DYN.compile(optimizer=opt, loss=tf.keras.losses.Huber(), metrics = ['mae'])
    else:
        DYN.compile(tf.keras.optimizers.SGD(learning_rate=l_r),
                    loss=MeanSquaredError(), run_eagerly=True)

    # TRAIN
    if not FLAGS['DYN']['control']:
        DYN.fit(zx_train, zy_train,
                epochs=N_epochs,
                shuffle=True,
                validation_data=(zx_val, zy_val),
                batch_size=N_batch)

    else:
        DYN.fit([zx_train, bx_train, by_train], zy_train,
                epochs=N_epochs,
                shuffle=True,
                validation_data=([zx_val, bx_val, by_val], zy_val),
                batch_size=N_batch)

    # LOAD WEIGHTS
    if flag_type == 'LSTM':
        with open(path_dyn[0], "rb") as fp:  # Unpickling
            LSTMs = pickle.load(fp)
        with open(path_dyn[1], "rb") as fp:  # Unpickling
            MLPs = pickle.load(fp)

        DYN.LSTMs.set_weights(LSTMs)
        DYN.MLPs.set_weights(MLPs)

    elif flag_type == 'NARX':
        with open(path_dyn[0], "rb") as fp:  # Unpickling
            MLP = pickle.load(fp)

        DYN.MLP.set_weights(MLP)

    return DYN

