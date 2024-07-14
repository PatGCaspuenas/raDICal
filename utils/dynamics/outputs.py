import numpy as np
import tensorflow as tf
import pickle
import os

from utils.data.transform_data import raw2dyn, CNNAE2raw
from utils.AEs.classes import CNN_VAE, C_CNN_AE

def energy_loss(input_img, decoded):
    return tf.keras.backend.sum(tf.keras.backend.square(input_img - decoded)) / tf.keras.backend.sum(tf.keras.backend.square(input_img))
def null_loss(input_img, decoded):
    return 0
def get_predicted_z(params, flags, DYN, Z, t, Znorm, u=0):

    flag_control_dyn = flags['dyn']['control']
    flag_control = flags['control']
    nt_pred = params['dyn']['nt_pred']

    if not flag_control:

        Zx_test, Zy_test, T = raw2dyn(t, Z, params, flag_control, flag_train=0)
        Zx_test, Zy_test = Zx_test / Znorm, Zy_test / Znorm

    else:

        Zx_test, Zy_test, Ux_test, Uy_test, T = raw2dyn(t, Z, params, flag_control, flag_train=0, u=u)
        Zx_test, Zy_test = Zx_test / Znorm, Zy_test / Znorm

    if flag_control_dyn:
        Zy_test_dyn = DYN.predict([Zx_test, Ux_test, Uy_test], nt_pred)
    else:
        Zy_test_dyn = DYN.predict(Zx_test, nt_pred)

    Zx_test, Zy_test, Zy_test_dyn = Zx_test * Znorm, Zy_test * Znorm, Zy_test_dyn * Znorm

    if not flag_control:
        return Zx_test, Zy_test, Zy_test_dyn, T
    else:
        return Zx_test, Zy_test, Zy_test_dyn, Ux_test, Uy_test, T

def load_model_AE(params, flags, paths):

    # FLAGS
    flag_AE = flags['AE']
    flag_control = flags['control']

    # PATHS
    path_AE = paths['model']

    # DEFINE AE TYPE & LOAD WEIGHTS
    if flag_AE == 'CNN-VAE':
        AE = CNN_VAE(params, flags)
    elif flag_AE == 'C-CNN-AE':
        AE = C_CNN_AE(params, flags)

    # LOSS
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    if flag_AE=='CNN-VAE':
        AE.compile(optimizer=opt, loss=null_loss, metrics=[energy_loss])
    else:
        AE.compile(optimizer=opt, loss='mse', metrics=[energy_loss])
    # INPUT
    if flag_control:
        input_train = [np.ones((32, AE.n, AE.m, AE.k)), tf.convert_to_tensor(np.zeros((32, 3)))]
        input_val =   [np.ones((32, AE.n, AE.m, AE.k)), tf.convert_to_tensor(np.zeros((32, 3)))]
    else:
        input_train = np.ones((32, AE.n, AE.m, AE.k))
        input_val =   np.ones((32, AE.n, AE.m, AE.k))

    # FIT
    AE.fit(input_train, np.ones((32, AE.n, AE.m, AE.k)),
                                     epochs=1,
                                     shuffle=False,
                                     validation_data=(input_val, np.ones((32, AE.n, AE.m, AE.k))),
                                     batch_size=32)
    cwd = os.getcwd()
    with open(cwd + r'/MODELS/encoder_' + path_AE, "rb") as fp:  # Unpickling
        enc = pickle.load(fp)
    with open(cwd + r'/MODELS/decoder_' + path_AE, "rb") as fp:  # Unpickling
        dec = pickle.load(fp)

    AE.encoder.set_weights(enc)
    AE.decoder.set_weights(dec)

    return AE

def get_predicted_flow(params, flags, paths, Zy, Zy_dyn, Uy=0):

    # PARAMS
    nW = np.shape(Zy)[0]
    nt = np.shape(Zy)[1]

    # FLAGS
    flag_control = flags['control']

    # LOAD MODEL
    AE = load_model_AE(params, flags, paths)

    # PREPARE FLOW
    m = AE.m
    n = AE.n
    k = AE.k

    D, Dr = np.zeros((2, nW, m * n * k, nt))

    # DECODE GROUND TRUTH & PREDICTED
    for w in range(nW):
        if flag_control:
            y_test = AE.decoder((tf.keras.layers.Concatenate(axis=1)([Zy[w, :, :], tf.convert_to_tensor(Uy[w, :, :])])))
            yr_test = AE.decoder((tf.keras.layers.Concatenate(axis=1)([Zy_dyn[w, :, :], tf.convert_to_tensor(Uy[w, :, :])])))
        else:
            y_test = AE.decoder(Zy[w, :, :])
            yr_test = AE.decoder(Zy_dyn[w, :, :])

        D[w, :, :] = CNNAE2raw(y_test)
        Dr[w, :, :] = CNNAE2raw(yr_test)

    return D, Dr


