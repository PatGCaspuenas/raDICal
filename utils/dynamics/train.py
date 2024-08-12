# PACKAGES
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping

# LOCAL FUNCTIONS
from utils.dynamics.classes import NARX, LSTM
from utils.data.transformer import raw2dyn
from utils.data.logger import MyLogger
from utils.modelling.custom_losses import MeanSquaredError

def train_dyn(PARAMS, FLAGS, z, t, logging, b=0):
    """
    Trains dynamical predictor model

    :param PARAMS: dictionary of parameters
    :param FLAGS: dictionary of flags
    :param z: latent space (N_t, N_z)
    :param t: time vector (N_t, 1)
    :param logging: logger object
    :param b: control vector, if needed (N_t, N_c)
    :return: dynamical predictor class model
    """

    # PARAMETERS
    l_r = PARAMS['DYN']['l_r']
    N_epochs = PARAMS['DYN']['N_epochs']
    N_batch = PARAMS['DYN']['N_batch']

    logger = MyLogger(logging, N_epochs)
    ES = EarlyStopping(monitor="val_loss", patience=100)

    # FLAGS
    flag_control = FLAGS['DYN']['control']
    flag_type = FLAGS['DYN']['type']
    flag_opt = FLAGS['DYN']['optimizer']
    flag_loss = FLAGS['DYN']['loss']


    # GENERATE WINDOW PREDICTIONS
    if not flag_control:
        zx_train, zy_train, zx_val, zy_val = raw2dyn(t, z, PARAMS, flag_control)
    else:
        zx_train, zy_train, zx_val, zy_val, bx_train, by_train, bx_val, by_val = raw2dyn(t, z, PARAMS, flag_control, b=b)

    N_w_train, N_w_val = np.shape(zx_train)[0], np.shape(zx_val)[0]
    logging.info(f'{N_w_train} training windows, {N_w_val} validation windows')

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
    logging.info(f'{flag_type} compilation COMPLETED with {flag_opt} optimizer and {flag_loss} loss...')

    # TRAIN
    if not FLAGS['DYN']['control']:
        DYN.fit(zx_train, zy_train,
                epochs=N_epochs,
                shuffle=True,
                validation_data=(zx_val, zy_val),
                batch_size=N_batch,
                verbose=2,
                callbacks=[logger, ES])

    else:
        DYN.fit([zx_train, bx_train, by_train], zy_train,
                epochs=N_epochs,
                shuffle=True,
                validation_data=([zx_val, bx_val, by_val], zy_val),
                batch_size=N_batch,
                verbose=2,
                callbacks=[logger, ES])
    logging.info(f'{flag_type} training process COMPLETED...')

    # Write to logger the NN summary structure
    if flag_type == 'NARX':
        stringlist = []
        DYN.MLP.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)

        logging.info(short_model_summary)

    else:
        stringlist = []
        DYN.lstm.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)

        logging.info(short_model_summary)

        stringlist = []
        DYN.predictor.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)

        logging.info(short_model_summary)

    return DYN





