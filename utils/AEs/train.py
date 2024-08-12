# PACKAGES
import numpy as np
import tensorflow as tf
import random
from keras.callbacks import EarlyStopping

# LOCAL FUNCTIONS
from utils.AEs.classes import CNN_VAE, CNN_HAE, C_CNN_AE, MD_CNN_AE
from utils.data.transformer import raw2CNNAE
from utils.data.logger import MyLogger
from utils.modelling.custom_losses import null_loss, energy_loss


def train_AE(PARAMS, FLAGS, grid, D, logging, b=0):
    """
    Trains and compiles AE

    :param PARAMS: dictionary of parameters
    :param FLAGS: dictionary of flags
    :param grid: dictionary containing X,Y and mask grids
    :param D: snapshot matrix (train + val set)
    :param logging: logger object
    :param b: control vector (if any)
    :return: AE class model
    """

    # FLAGS
    flag_AE = FLAGS['AE']['type']
    flag_control = FLAGS['AE']['control']

    # PARAMETERS
    N_z = PARAMS['AE']['N_z']
    N_epochs = PARAMS['AE']['N_epochs']
    N_batch = PARAMS['AE']['N_batch']
    l_r = PARAMS['AE']['l_r']

    # TRAINING AND VALIDATION DATA
    X_train, X_val, b_train, b_val = raw2CNNAE(grid, D, flag_train=1, flag_control=FLAGS['control'], u=b)
    N_t_train = np.shape(X_train)[0]
    N_t_val = np.shape(X_val)[0]
    logging.info(f'{N_t_train} training snapshots, {N_t_val} validation snapshots')

    # SHUFFLE TRAIN AND VALIDATION SETS IN SAME WAY
    i_train = [*range(np.shape(X_train)[0])]
    random.shuffle(i_train)

    i_val = [*range(np.shape(X_val)[0])]
    random.shuffle(i_val)
    logger = MyLogger(logging, N_epochs)
    ES = EarlyStopping(monitor="val_energy_loss", patience=50)

    # DEFINE AE TYPE
    if flag_AE == 'CNN-VAE':
        AE = CNN_VAE(PARAMS, FLAGS)
    elif flag_AE == 'MD-CNN-AE':
        AE = MD_CNN_AE(PARAMS, FLAGS)
    elif flag_AE == 'C-CNN-AE':
        AE = C_CNN_AE(PARAMS, FLAGS)


    if flag_AE=='CNN-HAE':

        AE = {}
        for i in range(N_z):

            # CREATE AE
            AE['m' + str(i + 1)] = CNN_HAE(PARAMS, FLAGS)
            opt = tf.keras.optimizers.Adam(learning_rate=l_r)

            # LOSS
            AE['m' + str(i + 1)].compile(optimizer=opt, loss='mse', metrics=[energy_loss])
            logging.info(f'{flag_AE} compilation COMPLETED...')

            # FIT
            if i == 0:

                AE['m' + str(i + 1)].fit([X_train[i_train, :, :, :]], X_train[i_train, :, :, :],
                                         epochs=N_epochs,
                                         shuffle=False,
                                         validation_data=([X_val[i_val, :, :, :]], X_val[i_val, :, :, :]),
                                         N_batch=N_batch,
                                         verbose=2,
                                         callbacks=[logger, ES])
                z_train = AE['m' + str(i + 1)].get_latent_vector(X_train)
                z_val = AE['m' + str(i + 1)].get_latent_vector(X_val)

            else:

                AE['m' + str(i + 1)].fit([X_train[i_train, :, :, :], tf.convert_to_tensor(z_train.numpy()[i_train, :])],
                                         X_train[i_train, :, :, :],
                                         epochs=N_epochs,
                                         shuffle=False,
                                         validation_data=(
                                         [X_val[i_val, :, :, :], tf.convert_to_tensor(z_val.numpy()[i_val, :])],
                                         X_val[i_val, :, :, :]),
                                         N_batch=N_batch,
                                         verbose=2,
                                         callbacks=[logger, ES])
                z_train = tf.keras.layers.Concatenate(axis=1)([z_train, AE['m' + str(i + 1)].get_latent_vector(X_train)])
                z_val = tf.keras.layers.Concatenate(axis=1)([z_val, AE['m' + str(i + 1)].get_latent_vector(X_val)])

    else:

        # LOSS
        opt = tf.keras.optimizers.Adam(learning_rate=l_r)
        if (flag_AE=='CNN-VAE') or (flag_AE=='C-CNN-AE-c'):
            AE.compile(optimizer=opt, loss=null_loss, metrics=[energy_loss, 'mse'])
        else:
            AE.compile(optimizer=opt, loss='mse', metrics=[energy_loss])
        logging.info(f'{flag_AE} compilation COMPLETED...')

        # INPUT
        if flag_control:
            input_train = [X_train[i_train,:,:,:], tf.convert_to_tensor(b_train[i_train,:])]
            input_val =   [X_val[i_val, :, :, :], tf.convert_to_tensor(b_val[i_val, :])]
        else:
            input_train = X_train[i_train,:,:,:]
            input_val =   X_val[i_val, :, :, :]

        # FIT
        AE.fit(input_train, X_train[i_train,:,:,:],
                                         epochs=N_epochs,
                                         shuffle=False,
                                         validation_data=(input_val, X_val[i_val,:,:,:]),
                                         batch_size=N_batch,
                                         verbose=2,
                                         callbacks=[logger, ES])

    logging.info(f'{flag_AE} training process COMPLETED...')
    return AE



