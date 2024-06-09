# PACKAGES
import numpy as np
import tensorflow as tf
import random
from keras.callbacks import Callback, EarlyStopping
from timeit import default_timer as timer

# LOCAL FUNCTIONS
from utils.AEs.classes import CNN_VAE, CNN_HAE, C_CNN_AE, MD_CNN_AE
from utils.data.transform_data import raw2CNNAE

class MyLogger(Callback):
    def __init__(self,logging, epochs):
        super(MyLogger, self).__init__()
        self.logging = logging
        self.n_epoch = epochs
    def on_train_batch_end(self, batch, logs=None):
        self.batch_n += 1
    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = timer()
        self.batch_n = 0
    def on_epoch_end(self, epoch, logs=None):
        self.logging.info(f'Epoch {epoch}/{self.n_epoch} - {self.batch_n}/{self.batch_n} - {(timer()-self.starttime)}s - {logs}')

# def energy_loss(input_img, decoded):
#     return tf.keras.ops.sum(tf.keras.ops.square(input_img - decoded)) / tf.keras.ops.sum(tf.keras.ops.square(input_img))

def energy_loss(input_img, decoded):
    return tf.keras.backend.sum(tf.keras.backend.square(input_img - decoded)) / tf.keras.backend.sum(tf.keras.backend.square(input_img))

def train_AE(params, flags, grid, Ddt, logging, b=0):

    # FLAGS
    flag_AE = flags['AE']
    flag_control = flags['control']

    # PARAMETERS
    nr = params['AE']['nr']
    n_epochs = params['AE']['n_epochs']
    batch_size = params['AE']['batch_size']
    lr = params['AE']['lr']

    # TRAINING AND VALIDATION DATA
    X_train, X_val, b_train, b_val = raw2CNNAE(grid, Ddt, flag_split=1, flag_control=flags['control'], u=b)
    nt_train = np.shape(X_train)[0]
    nt_val = np.shape(X_val)[0]
    logging.info(f'{nt_train} training snapshots, {nt_val} validation snapshots')

    # SHUFFLE TRAIN AND VALIDATION SETS IN SAME WAY
    i_train = [*range(np.shape(X_train)[0])]
    random.shuffle(i_train)

    i_val = [*range(np.shape(X_val)[0])]
    random.shuffle(i_val)
    logger = MyLogger(logging, n_epochs)
    # ES = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=50)

    # DEFINE AE TYPE
    if flag_AE == 'CNN-VAE':
        AE = CNN_VAE(params, flags)
    elif flag_AE == 'MD-CNN-AE':
        AE = MD_CNN_AE(params, flags)
    elif flag_AE == 'C-CNN-AE':
        AE = C_CNN_AE(params, flags)


    if flag_AE=='CNN-HAE':

        AE = {}
        for i in range(nr):

            # CREATE AE
            AE['m' + str(i + 1)] = CNN_HAE(params, flags)
            opt = tf.keras.optimizers.Adam(learning_rate=lr)

            # LOSS
            AE['m' + str(i + 1)].compile(optimizer=opt, loss='mse', metrics=[energy_loss])

            # FIT
            if i == 0:

                AE['m' + str(i + 1)].fit([X_train[i_train, :, :, :]], X_train[i_train, :, :, :],
                                         epochs=n_epochs,
                                         shuffle=False,
                                         validation_data=([X_val[i_val, :, :, :]], X_val[i_val, :, :, :]),
                                         batch_size=batch_size,
                                         verbose=2,
                                         callbacks=[logger])
                z_train = AE['m' + str(i + 1)].get_latent_vector(X_train)
                z_val = AE['m' + str(i + 1)].get_latent_vector(X_val)

            else:

                AE['m' + str(i + 1)].fit([X_train[i_train, :, :, :], tf.convert_to_tensor(z_train.numpy()[i_train, :])],
                                         X_train[i_train, :, :, :],
                                         epochs=n_epochs,
                                         shuffle=False,
                                         validation_data=(
                                         [X_val[i_val, :, :, :], tf.convert_to_tensor(z_val.numpy()[i_val, :])],
                                         X_val[i_val, :, :, :]),
                                         batch_size=batch_size,
                                         verbose=2,
                                         callbacks=[logger])
                z_train = tf.keras.layers.Concatenate(axis=1)([z_train, AE['m' + str(i + 1)].get_latent_vector(X_train)])
                z_val = tf.keras.layers.Concatenate(axis=1)([z_val, AE['m' + str(i + 1)].get_latent_vector(X_val)])

    else:

        # LOSS
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        AE.compile(optimizer=opt, loss='mse', metrics=[energy_loss])

        # INPUT
        if flag_control:
            input_train = [X_train[i_train,:,:,:], tf.convert_to_tensor(b_train[i_train,:])]
            input_val =   [X_val[i_val, :, :, :], tf.convert_to_tensor(b_val[i_val, :])]
        else:
            input_train = X_train[i_train,:,:,:]
            input_val =   X_val[i_val, :, :, :]

        # FIT
        AE.fit(input_train, X_train[i_train,:,:,:],
                                         epochs=n_epochs,
                                         shuffle=False,
                                         validation_data=(input_val, X_val[i_val,:,:,:]),
                                         batch_size=batch_size,
                                         verbose=2,
                                         callbacks=[logger])

    return AE



