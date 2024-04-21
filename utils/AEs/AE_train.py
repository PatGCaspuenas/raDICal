import numpy as np
import tensorflow as tf
import random

#from utils.AEs.AE_classes import MD_CNN_AE
from utils.AEs.AE_classes_complex import CNN_VAE, CNN_HAE, C_CNN_AE, MD_CNN_AE

# FUNCTIONS

def energy_loss(input_img, decoded):
    return tf.keras.ops.sum(tf.keras.ops.square(input_img - decoded)) / tf.keras.ops.sum(tf.keras.ops.square(input_img))
def train_CNNHAE(n_epochs, ksize, psize, ptypepool, nstrides, act, nr, X_train, X_val, X_test,):

    AE = {}
    # Create same shuffling for each AE
    i_train = [*range(np.shape(X_train)[0])]
    random.shuffle(i_train)

    i_val = [*range(np.shape(X_val)[0])]
    random.shuffle(i_val)

    for i in range(nr):
        AE['m'+str(i+1)] = CNN_HAE(ksize, psize, ptypepool, nstrides, act, nr, X_val)

        AE['m' + str(i + 1)].compile(optimizer='adam', loss=energy_loss)

        if i == 0:
            AE['m' + str(i + 1)].fit([X_train[i_train,:,:,:]], X_train[i_train,:,:,:],
                                     epochs=n_epochs,
                                     shuffle=False,
                                     validation_data=([X_val[i_val,:,:,:]], X_val[i_val,:,:,:]))
            lat_vector_train = AE['m' + str(i + 1)].get_latent_vector(X_train)
            lat_vector_val = AE['m' + str(i + 1)].get_latent_vector(X_val)
            lat_vector_test = AE['m' + str(i + 1)].get_latent_vector(X_test)
        else:
            AE['m' + str(i + 1)].fit([X_train[i_train,:,:,:], tf.convert_to_tensor(lat_vector_train.numpy()[i_train,:])], X_train[i_train,:,:,:],
                                     epochs=n_epochs,
                                     shuffle=False,
                                     validation_data=([X_val[i_val,:,:,:], tf.convert_to_tensor(lat_vector_val.numpy()[i_val,:])], X_val[i_val,:,:,:]),
                                     batch_size=16,
                                     verbose=2)
            lat_vector_train = tf.keras.layers.Concatenate(axis=1)([lat_vector_train, AE['m' + str(i + 1)].get_latent_veector(X_train)])
            lat_vector_val = tf.keras.layers.Concatenate(axis=1)(
                [lat_vector_val, AE['m' + str(i + 1)].get_latent_vector(X_val)])
            lat_vector_test = tf.keras.layers.Concatenate(axis=1)(
                [lat_vector_test, AE['m' + str(i + 1)].get_latent_vector(X_test)])
        tf.keras.backend.clear_session()

    return AE, lat_vector_test

def train_CNNVAE(beta, n_epochs, ksize, psize, ptypepool, nstrides, act, nr, X_train, X_val, X_test):

    # Create same shuffling for each AE
    i_train = [*range(np.shape(X_train)[0])]
    random.shuffle(i_train)

    i_val = [*range(np.shape(X_val)[0])]
    random.shuffle(i_val)

    AE = CNN_VAE(beta, ksize, psize, ptypepool, nstrides, act, nr, X_val)

    #step = tf.Variable(0, trainable=False)
    boundaries = [500, 100, 100]
    values = [1e-3, 1e-4, 1e-5, 1e-6]
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    AE.compile(optimizer=opt, loss='mse', metrics=[energy_loss])


    AE.fit(X_train[i_train,:,:,:], X_train[i_train,:,:,:],
                                     epochs=n_epochs,
                                     shuffle=False,
                                     validation_data=(X_val[i_val,:,:,:], X_val[i_val,:,:,:]),
                                     batch_size=32,
                                     verbose=2)
    tf.keras.backend.clear_session()
    z_test = AE.get_latent_vector(X_test)

    return AE, z_test

def train_MDCNNAE(n_epochs, ksize, psize, ptypepool, nstrides, act, nr, X_train, X_val):

    # Create same shuffling for each AE
    i_train = [*range(np.shape(X_train)[0])]
    random.shuffle(i_train)

    i_val = [*range(np.shape(X_val)[0])]
    random.shuffle(i_val)

    AE = MD_CNN_AE(ksize, psize, ptypepool, nstrides, act, nr, X_val)
    AE.compile(optimizer='adam', loss='mse')
    AE.fit(X_train[i_train,:,:,:], X_train[i_train,:,:,:],
           epochs=n_epochs,
           shuffle=False,
           validation_data=(X_val[i_val,:,:,:], X_val[i_val,:,:,:]),
           batch_size=16,
           verbose=2)
    tf.keras.backend.clear_session()

    return AE

def train_CCNNAE(n_epochs, ksize, psize, ptypepool, nstrides, act, nr, X_train, X_val, X_test):

    # Create same shuffling for each AE
    i_train = [*range(np.shape(X_train)[0])]
    random.shuffle(i_train)

    i_val = [*range(np.shape(X_val)[0])]
    random.shuffle(i_val)

    AE = C_CNN_AE(ksize, psize, ptypepool, nstrides, act, nr, X_val)
    AE.compile(optimizer='Adam', loss=energy_loss)


    AE.fit(X_train[i_train,:,:,:], X_train[i_train,:,:,:],
                                     epochs=n_epochs,
                                     shuffle=False,
                                     validation_data=(X_val[i_val,:,:,:], X_val[i_val,:,:,:]),
                                     batch_size=32,
                                     verbose=2)
    tf.keras.backend.clear_session()
    z_test = AE.get_latent_vector(X_test)

    return AE, z_test