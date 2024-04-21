# PACKAGES
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.regularizers import l2
import tensorflow as tf
class MD_CNN_AE(Model):

    def __init__(self, ksize, psize, ptypepool, nstrides, act, nr, input_img):

        # Call parent constructor
        super(MD_CNN_AE, self).__init__()

        # Define parameters
        self.ksize = ksize
        self.psize = psize
        self.ptypepool = ptypepool
        self.nstrides = nstrides
        self.nr = nr
        self.act = act

        self.n = np.shape(input_img)[1]
        self.m = np.shape(input_img)[2]
        self.k = np.shape(input_img)[3]

        self.encoder = self.create_encoder()
        for i in range(nr):
            setattr(self, 'decoder'+str(i+1), self.create_decoder())

    def create_encoder(self):

        m_sub = self.m
        n_sub = self.n

        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.layers.Input(shape=(self.n, self.m, self.k)))

        # 1st: conv & max pooling
        # encoder.add(tf.keras.layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))
        # encoder.add(tf.keras.layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        # n_sub = (n_sub) // self.nstrides
        # m_sub = (m_sub) // self.nstrides

        # 2nd: conv & max pooling
        encoder.add(layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 3rd: conv & max pooling
        encoder.add(layers.Conv2D(32, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 4th: conv & max pooling
        encoder.add(layers.Conv2D(64, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 5th: conv & max pooling
        encoder.add(layers.Conv2D(128, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 6th: conv & max pooling
        encoder.add(layers.Conv2D(256, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # fully connected
        encoder.add(layers.Reshape([m_sub * n_sub * 256]))
        encoder.add(layers.Dense(128, activation=self.act))

        encoder.add(layers.Dense(self.nr, activation=self.act))

        self.m_sub = m_sub
        self.n_sub = n_sub

        return encoder

    def create_decoder(self):

        m_sub = self.m_sub
        n_sub = self.n_sub

        decoder = tf.keras.Sequential()

        # fully connected
        decoder.add(layers.Dense(128, activation=self.act))

        decoder.add(layers.Dense(m_sub * n_sub * 256, activation=self.act))
        decoder.add(layers.Reshape([n_sub, m_sub, 256]))

        # 1st: upsampling & (7th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(256, self.ksize, activation=self.act, padding='same'))

        # 2nd: upsampling & (8th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(128, self.ksize, activation=self.act, padding='same'))

        # 3rd: upsampling & (9th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(64, self.ksize, activation=self.act, padding='same'))

        # 4th: upsampling & (10th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(32, self.ksize, activation=self.act, padding='same'))

        # 5th: upsampling & (11th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))

        # 6th: upsampling & (12th) conv
        # decoder.add(layers.UpSampling2D(self.psize))
        # decoder.add(layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))

        decoder.add(layers.Conv2D(self.k, self.ksize, activation=self.act, padding='same'))

        return decoder

    def extract_mode(self, input_img, ni):

        encoded = self.encoder(input_img)
        lat_vector = tf.keras.layers.Lambda(lambda x: x[:, ni-1:ni])(encoded)
        decoded = getattr(self, 'decoder'+str(ni))(lat_vector)

        return decoded

    def call(self, input_img):

        encoded = self.encoder(input_img)
        for i in range(self.nr):
            lat_vector = tf.keras.layers.Lambda(lambda x: x[:, i:i + 1])(encoded)
            decoded_sub = getattr(self, 'decoder'+str(i+1))(lat_vector)

            if i == 0:
                decoded = tf.keras.layers.Add()([decoded_sub])
            else:
                decoded = tf.keras.layers.Add()([decoded, decoded_sub])

        return decoded

class CNN_HAE(Model):

    def __init__(self, ksize, psize, ptypepool, nstrides, act, nr, input_img):

        # Call parent constructor
        super(CNN_HAE, self).__init__()

        # Define parameters
        self.ksize = ksize
        self.psize = psize
        self.ptypepool = ptypepool
        self.nstrides = nstrides
        self.nr = nr
        self.act = act

        self.n = np.shape(input_img)[1]
        self.m = np.shape(input_img)[2]
        self.k = np.shape(input_img)[3]

        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

    def create_encoder(self):

        m_sub = self.m
        n_sub = self.n

        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.layers.Input(shape=(self.n, self.m, self.k)))

        # 1st: conv & max pooling
        # encoder.add(tf.keras.layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))
        # encoder.add(tf.keras.layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        # n_sub = (n_sub) // self.nstrides
        # m_sub = (m_sub) // self.nstrides

        # 2nd: conv & max pooling
        encoder.add(layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 3rd: conv & max pooling
        encoder.add(layers.Conv2D(32, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 4th: conv & max pooling
        encoder.add(layers.Conv2D(64, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 5th: conv & max pooling
        encoder.add(layers.Conv2D(128, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 6th: conv & max pooling
        encoder.add(layers.Conv2D(256, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # fully connected
        encoder.add(layers.Reshape([m_sub * n_sub * 256]))
        encoder.add(layers.Dense(128, activation=self.act))

        encoder.add(layers.Dense(1, activation=self.act))

        self.m_sub = m_sub
        self.n_sub = n_sub

        return encoder

    def create_decoder(self):

        m_sub = self.m_sub
        n_sub = self.n_sub

        decoder = tf.keras.Sequential()

        # fully connected
        decoder.add(layers.Dense(128, activation=self.act))

        decoder.add(layers.Dense(m_sub * n_sub * 256, activation=self.act))
        decoder.add(layers.Reshape([n_sub, m_sub, 256]))

        # 1st: upsampling & (7th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(256, self.ksize, activation=self.act, padding='same'))

        # 2nd: upsampling & (8th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(128, self.ksize, activation=self.act, padding='same'))

        # 3rd: upsampling & (9th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(64, self.ksize, activation=self.act, padding='same'))

        # 4th: upsampling & (10th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(32, self.ksize, activation=self.act, padding='same'))

        # 5th: upsampling & (11th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))

        # 6th: upsampling & (12th) conv
        # decoder.add(layers.UpSampling2D(self.psize))
        # decoder.add(layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))

        decoder.add(layers.Conv2D(self.k, self.ksize, activation=self.act, padding='same'))

        return decoder

    def get_reconstruction(self, input):

        input_img = input[0]
        if len(input) == 2:
            latent_vector = input[1]

        encoded = self.encoder(input_img)
        if len(input) == 2:
            decoded = self.decoder(tf.keras.layers.Concatenate(axis=1)([latent_vector, encoded]))
        else:
            decoded = self.decoder(encoded)

        return decoded

    def get_latent_vector(self, input_img):

        encoded = self.encoder(input_img)
        return encoded

    def call(self, input):

        input_img = input[0]
        if len(input) == 2:
            latent_vector = input[1]

        encoded = self.encoder(input_img)
        if len(input) == 2:
            decoded = self.decoder(tf.keras.layers.Concatenate(axis=1)([latent_vector, encoded]))
        else:
            decoded = self.decoder(encoded)
        return decoded

class CNN_VAE(Model):

    def __init__(self, beta, ksize, psize, ptypepool, nstrides, act, nr, input_img):

        # Call parent constructor
        super(CNN_VAE, self).__init__()

        # Define parameters
        self.beta = beta
        self.ksize = ksize
        self.psize = psize
        self.ptypepool = ptypepool
        self.nstrides = nstrides
        self.nr = nr
        self.act = act

        self.n = np.shape(input_img)[1]
        self.m = np.shape(input_img)[2]
        self.k = np.shape(input_img)[3]

        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

    def create_encoder(self):

        m_sub = self.m
        n_sub = self.n

        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.layers.Input(shape=(self.n, self.m, self.k)))

        # 1st: conv & max pooling
        # encoder.add(tf.keras.layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))
        # encoder.add(tf.keras.layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        # n_sub = (n_sub) // self.nstrides
        # m_sub = (m_sub) // self.nstrides

        # 2nd: conv & max pooling
        encoder.add(layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 3rd: conv & max pooling
        encoder.add(layers.Conv2D(32, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 4th: conv & max pooling
        encoder.add(layers.Conv2D(64, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 5th: conv & max pooling
        encoder.add(layers.Conv2D(128, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 6th: conv & max pooling
        encoder.add(layers.Conv2D(256, self.ksize, activation=self.act, padding='same'))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # fully connected
        encoder.add(layers.Reshape([m_sub * n_sub * 256]))
        encoder.add(layers.Dense(128, activation=self.act))

        encoder.add(layers.Dense(self.nr + self.nr, activation=self.act))

        self.m_sub = m_sub
        self.n_sub = n_sub

        return encoder

    def create_decoder(self):

        m_sub = self.m_sub
        n_sub = self.n_sub

        decoder = tf.keras.Sequential()

        # fully connected
        decoder.add(layers.Dense(128, activation=self.act))

        decoder.add(layers.Dense(m_sub * n_sub * 256, activation=self.act))
        decoder.add(layers.Reshape([n_sub, m_sub, 256]))

        # 1st: upsampling & (7th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(256, self.ksize, activation=self.act, padding='same'))

        # 2nd: upsampling & (8th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(128, self.ksize, activation=self.act, padding='same'))

        # 3rd: upsampling & (9th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(64, self.ksize, activation=self.act, padding='same'))

        # 4th: upsampling & (10th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(32, self.ksize, activation=self.act, padding='same'))

        # 5th: upsampling & (11th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))

        # 6th: upsampling & (12th) conv
        # decoder.add(layers.UpSampling2D(self.psize))
        # decoder.add(layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))

        decoder.add(layers.Conv2D(self.k, self.ksize, activation=self.act, padding='same'))

        return decoder

    def get_latent_vector(self, input_img):

        encoded = self.encoder(input_img)
        z_mean, z_logvar = tf.split(encoded, num_or_size_splits=2, axis=1)
        z = self.sampling(z_mean, z_logvar)
        return z
    def sampling(self, z_mean, z_log_sigma):
        epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], self.nr),
                                  mean=0., stddev=1.0)
        return z_mean + tf.keras.backend.exp(z_log_sigma) * epsilon

    def loss_fn(self, input_img, decoded, z_logvar, z_mean):
        rec_loss = tf.keras.backend.sum(tf.keras.backend.square(input_img - decoded)) / tf.keras.backend.sum(tf.keras.backend.square(input_img))
        kl_loss = 1 + z_logvar - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_logvar)
        kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = rec_loss + tf.keras.backend.mean(self.beta * kl_loss)
        return rec_loss, kl_loss, vae_loss

    def call(self, input_img):

        encoded = self.encoder(input_img)
        z_mean, z_logvar = tf.split(encoded, num_or_size_splits=2, axis=1)
        z = self.sampling(z_mean, z_logvar)

        decoded = self.decoder(z)

        rec_loss, kl_loss, vae_loss = self.loss_fn(input_img, decoded, z_logvar, z_mean)
        self.add_loss(vae_loss)
        # self.add_metric(rec_loss, aggregation='mean')
        # self.add_metric(kl_loss, name='kl_loss', aggregation='mean')

        return decoded

class C_CNN_AE(Model):

    def __init__(self, ksize, psize, ptypepool, nstrides, act, nr, input_img):

        # Call parent constructor
        super(C_CNN_AE, self).__init__()

        # Define parameters
        self.ksize = ksize
        self.psize = psize
        self.ptypepool = ptypepool
        self.nstrides = nstrides
        self.act = act
        self.nr = nr

        self.n = np.shape(input_img)[1]
        self.m = np.shape(input_img)[2]
        self.k = np.shape(input_img)[3]

        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

    def create_encoder(self):

        m_sub = self.m
        n_sub = self.n

        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.layers.Input(shape=(self.n, self.m, self.k)))

        # 1st: conv & max pooling
        # encoder.add(tf.keras.layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))
        # encoder.add(tf.keras.layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        # n_sub = (n_sub) // self.nstrides
        # m_sub = (m_sub) // self.nstrides

        # 2nd: conv & max pooling
        encoder.add(layers.Conv2D(16, self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(0), bias_regularizer=l2(0)))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 3rd: conv & max pooling
        encoder.add(layers.Conv2D(32, self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(0), bias_regularizer=l2(0)))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 4th: conv & max pooling
        encoder.add(layers.Conv2D(64, self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(0), bias_regularizer=l2(0)))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 5th: conv & max pooling
        encoder.add(layers.Conv2D(128, self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(0), bias_regularizer=l2(0)))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # 6th: conv & max pooling
        encoder.add(layers.Conv2D(256, self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(0), bias_regularizer=l2(0)))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # fully connected
        encoder.add(layers.Reshape([m_sub * n_sub * 256]))
        encoder.add(layers.Dense(128, activation=self.act, kernel_regularizer=l2(0), bias_regularizer=l2(0)))

        encoder.add(layers.Dense(self.nr, activation=self.act, kernel_regularizer=l2(0), bias_regularizer=l2(0)))

        self.m_sub = m_sub
        self.n_sub = n_sub

        return encoder

    def create_decoder(self):

        m_sub = self.m_sub
        n_sub = self.n_sub

        decoder = tf.keras.Sequential()

        # fully connected
        decoder.add(layers.Dense(128, activation=self.act, kernel_regularizer=l2(0), bias_regularizer=l2(0)))

        decoder.add(layers.Dense(m_sub * n_sub * 256, activation=self.act, kernel_regularizer=l2(0), bias_regularizer=l2(0)))
        decoder.add(layers.Reshape([n_sub, m_sub, 256]))

        # 1st: upsampling & (7th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(256, self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(0), bias_regularizer=l2(0)))

        # 2nd: upsampling & (8th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(128, self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(0), bias_regularizer=l2(0)))

        # 3rd: upsampling & (9th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(64, self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(0), bias_regularizer=l2(0)))

        # 4th: upsampling & (10th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(32, self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(0), bias_regularizer=l2(0)))

        # 5th: upsampling & (11th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(16, self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(0), bias_regularizer=l2(0)))

        # 6th: upsampling & (12th) conv
        # decoder.add(layers.UpSampling2D(self.psize))
        # decoder.add(layers.Conv2D(16, self.ksize, activation=self.act, padding='same'))

        decoder.add(layers.Conv2D(self.k, self.ksize, activation=self.act, padding='same'))

        return decoder

    def get_latent_vector(self, input_img):

        encoded = self.encoder(input_img)

        return encoded


    def call(self, input_img):

        decoded = self.decoder(self.encoder(input_img))

        return decoded
