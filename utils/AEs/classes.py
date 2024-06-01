# PACKAGES
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.regularizers import l2
import tensorflow as tf

# PARENT CLASS - STRUCTURE AND UTILS
class AE(object):

    def __init__(self, params, flags):

        super(AE, self).__init__()

        # AE parameters
        self.beta = params['AE']['beta']           # CNN-VAE reg parameter
        self.ksize = params['AE']['ksize']         # Kernel size of convolutional layers
        self.psize = params['AE']['psize']         # Kernel size of pooling layers
        self.ptypepool = params['AE']['ptypepool'] # Type of pooling padding (same or valid)
        self.nstrides = params['AE']['nstrides']   # Number of strides in pooling
        self.act = params['AE']['act']             # Activation function
        self.reg_k = params['AE']['reg_k']         # Regularization kernel
        self.reg_b = params['AE']['reg_b']         # Regularization bias
        self.nr = params['AE']['nr']               # Latent space dimensions

        # Data parameters
        self.n = params['flow']['n'] # Columns in grid
        self.m = params['flow']['m'] # Rows in grid
        self.k = params['flow']['k'] # Number of dimensions

        # Flags
        self.flag_AE = flags['AE']           # AE type (C-CNN-AE, MD-CNN-AE, CNN-HAE, CNN-VAE)
        self.flag_struct = flags['struct']   # AE structure type (simple, medium, complex)
        self.flag_flow = flags['flow']       # Flow type (SC, FP)
        self.flag_control = flags['control'] # With (1) or without (0) control

        # Latent space at the end of encoder
        if (self.flag_AE=='CNN-HAE'):
            self.ne = 1
        elif (self.flag_AE=='C-CNN-AE') or (self.flag_AE=='MD-CNN-AE'):
            self.ne = self.nr
        elif (self.flag_AE=='CNN-VAE'):
            self.ne = self.nr + self.nr

        # Define # filters in layer sequential
        if (self.flag_struct=='simple'):
            self.filter_seq_e = [16,  8,  8,  8,   4,   4]
            self.filter_seq_d = [ 4,  4,  8,  8,   8,  16]
        elif (self.flag_struct=='medium'):
            self.filter_seq_e = [ 4,  4,  8, 16,  32,  64, 32]
            self.filter_seq_d = [128, 256, 256, 128, 64, 32, 16, 16]
        elif (self.flag_struct=='complex'):
            self.filter_seq_e = [16, 16, 32, 64, 128, 256, 128]
            self.filter_seq_d = [64, 128, 128, 64, 32, 16, 8, 8]

        # Create encoder and decoder(s)
        self.encoder = self.create_encoder()
        if self.flag_AE == 'MD-CNN-AE':
            for i in range(self.nr):
                setattr(self, 'decoder' + str(i + 1), self.create_decoder())
        else:
            self.decoder = self.create_decoder()

    def create_encoder(self):

        m_sub = self.m
        n_sub = self.n
        i_seq = 0

        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.layers.Input(shape=(self.n, self.m, self.k)))

        # 1st: conv & max pooling (if m x n = 392 x 196)
        if self.n == 384:
            encoder.add(tf.keras.layers.Conv2D(self.filter_seq_e[i_seq], self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
            encoder.add(tf.keras.layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
            n_sub = (n_sub) // self.nstrides
            m_sub = (m_sub) // self.nstrides
            i_seq += 1

        # 2nd: conv & max pooling
        encoder.add(layers.Conv2D(self.filter_seq_e[i_seq], self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides
        i_seq += 1

        # 3rd: conv & max pooling
        encoder.add(layers.Conv2D(self.filter_seq_e[i_seq], self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides
        i_seq += 1

        # 4th: conv & max pooling
        encoder.add(layers.Conv2D(self.filter_seq_e[i_seq], self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides
        i_seq += 1

        # 5th: conv & max pooling
        encoder.add(layers.Conv2D(self.filter_seq_e[i_seq], self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides
        i_seq += 1

        # 6th: conv & max pooling
        encoder.add(layers.Conv2D(self.filter_seq_e[i_seq], self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
        encoder.add(layers.MaxPooling2D(pool_size=self.psize, padding=self.ptypepool, strides=self.nstrides))
        n_sub = (n_sub) // self.nstrides
        m_sub = (m_sub) // self.nstrides

        # fully connected
        encoder.add(layers.Reshape([m_sub * n_sub * self.filter_seq_e[i_seq]]))
        i_seq += 1
        if self.flag_struct != 'simple':
            encoder.add(layers.Dense(self.filter_seq_e[i_seq], activation=self.act, kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))

        encoder.add(layers.Dense(self.ne, activation=self.act, kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))

        self.m_sub = m_sub
        self.n_sub = n_sub

        return encoder
    def create_decoder(self):

        m_sub = self.m_sub
        n_sub = self.n_sub
        i_seq = 0

        decoder = tf.keras.Sequential()

        # fully connected
        if self.flag_struct != 'simple':
            decoder.add(layers.Dense(m_sub * n_sub * self.filter_seq_d[i_seq], activation=self.act, kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
            i_seq += 1

        decoder.add(layers.Dense(m_sub * n_sub * self.filter_seq_d[i_seq], activation=self.act, kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
        decoder.add(layers.Reshape([n_sub, m_sub, self.filter_seq_d[i_seq]]))
        i_seq += 1

        # 1st: upsampling & (7th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(self.filter_seq_d[i_seq], self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
        i_seq += 1

        # 2nd: upsampling & (8th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(self.filter_seq_d[i_seq], self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
        i_seq += 1

        # 3rd: upsampling & (9th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(self.filter_seq_d[i_seq], self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
        i_seq += 1

        # 4th: upsampling & (10th) conv
        decoder.add(layers.UpSampling2D(self.psize))
        decoder.add(layers.Conv2D(self.filter_seq_d[i_seq], self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
        i_seq += 1

        # 5th: upsampling & (11th) conv
        if self.flag_struct != 'simple':
            decoder.add(layers.UpSampling2D(self.psize))
            decoder.add(layers.Conv2D(self.filter_seq_d[i_seq], self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))
            i_seq += 1

        # 6th: upsampling & (12th) conv  (if m x n = 392 x 196)
        if self.n == 384:
            decoder.add(layers.UpSampling2D(self.psize))
            decoder.add(layers.Conv2D(self.filter_seq_d[i_seq], self.ksize, activation=self.act, padding='same', kernel_regularizer=l2(self.reg_k), bias_regularizer=l2(self.reg_b)))

        if self.flag_struct == 'simple':
            decoder.add(layers.UpSampling2D(self.psize))

        decoder.add(layers.Conv2D(self.k, self.ksize, activation=self.act, padding='same'))

        return decoder


class MD_CNN_AE(AE, Model):

    def __init__(self, params, flags):

        # Call parent constructor
        super(MD_CNN_AE, self).__init__(params=params, flags=flags)


    def extract_mode(self, input_img, ni):

        encoded = self.encoder(input_img)
        lat_vector = tf.keras.layers.Lambda(lambda x: x[:, ni-1:ni])(encoded)
        decoded = getattr(self, 'decoder'+str(ni))(lat_vector)

        return decoded

    def get_latent_vector(self, input_img):

        z = self.encoder(input_img)

        return z

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

class CNN_HAE(AE, Model):

    def __init__(self, params, flags):

        # Call parent constructor
        super(CNN_HAE, self).__init__(params=params, flags=flags)

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

class CNN_VAE(AE, Model):

    def __init__(self, params, flags):

        # Call parent constructor
        super(CNN_VAE, self).__init__(params=params, flags=flags)


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

    def call(self, input):

        if not self.flag_control:
            input_img = input
        else:
            input_img = input[0]
            control_vector = input[1]

        encoded = self.encoder(input_img)
        z_mean, z_logvar = tf.split(encoded, num_or_size_splits=2, axis=1)
        z = self.sampling(z_mean, z_logvar)

        if not self.flag_control:
            decoded = self.decoder(z)
        else:
            decoded = self.decoder(tf.keras.layers.Concatenate(axis=1)([z, control_vector]))

        rec_loss, kl_loss, vae_loss = self.loss_fn(input_img, decoded, z_logvar, z_mean)
        self.add_loss(vae_loss)
        #self.add_metric(rec_loss)
        #self.add_metric(kl_loss)

        return decoded

class C_CNN_AE(AE, Model):

    def __init__(self, params, flags):

        # Call parent constructor
        super(C_CNN_AE, self).__init__(params=params, flags=flags)

    def get_latent_vector(self, input_img):

        encoded = self.encoder(input_img)

        return encoded


    def call(self, input):

        if not self.flag_control:
            decoded = self.decoder(self.encoder(input))
        else:
            input_img = input[0]
            control_vector = input[1]

            encoded = self.encoder(input_img)
            decoded = self.decoder(tf.keras.layers.Concatenate(axis=1)([encoded, control_vector]))

        return decoded


