# PACKAGES
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.regularizers import l2
import tensorflow as tf

# PARENT CLASS - STRUCTURE AND UTILS
class AE(object):
    """
    AE parent class, containing initialization of parameters, encoder and decoder structure and common utils
    """

    def __init__(self, PARAMS, FLAGS):

        super(AE, self).__init__()

        # AE parameters
        self.beta = PARAMS['AE']['beta']             # CNN-VAE reg parameter
        self.k_size = PARAMS['AE']['k_size']         # Kernel size of convolutional layers
        self.p_size = PARAMS['AE']['p_size']         # Kernel size of pooling layers
        self.p_pad = FLAGS['AE']['p_pad']            # Type of pooling padding (same or valid)
        self.p_strides = PARAMS['AE']['p_strides']   # Number of strides in pooling
        self.activation = FLAGS['AE']['activation']  # Activation function
        self.k_reg = PARAMS['AE']['k_reg']           # Regularization kernel
        self.k_b = PARAMS['AE']['k_b']               # Regularization bias
        self.drp_rate = PARAMS['AE']['drp_rate']     # Dropout rate after dense layers
        self.N_z = PARAMS['AE']['N_z']               # Latent space dimensions

        # Data parameters
        self.N_x =  PARAMS['FLOW']['N_x'] # Columns in grid
        self.N_y =  PARAMS['FLOW']['N_y'] # Rows in grid
        self.K =    PARAMS['FLOW']['K']   # Number of dimensions

        # FLAGS
        self.flag_AE = FLAGS['AE']["type"]              # AE type (C-CNN-AE, MD-CNN-AE, CNN-HAE, CNN-VAE)
        self.flag_struct = FLAGS['AE']['architecture']  # AE structure type (simple, medium, complex)
        self.flag_control = FLAGS['AE']['control']      # With (1) or without (0) control embedded in the latent space

        # Latent space at the end of encoder
        if (self.flag_AE=='CNN-HAE'):
            self.N_e = 1
        elif (self.flag_AE=='C-CNN-AE') or (self.flag_AE=='MD-CNN-AE'):
            self.N_e = self.N_z
        elif (self.flag_AE=='CNN-VAE'):
            self.N_e = self.N_z + self.N_z

        # Define # filters in layer sequential
        if (self.flag_struct=='simple'):
            self.filter_seq_e = [16,  8,  8,  8,   4,   4]
            self.filter_seq_d = [ 4,  4,  8,  8,   8,  16]
        elif (self.flag_struct=='medium'):
            self.filter_seq_e = [ 4,  4,  8, 16,  32,  64, 32]
            self.filter_seq_d = [32, 64, 32, 16, 8, 4, 4]
        elif (self.flag_struct=='complex'):
            self.filter_seq_e = [16, 16, 32, 64, 128, 256, 128]
            self.filter_seq_d = [128, 256, 256, 128, 64, 32, 16, 16]

        # Create encoder and decoder(s)
        self.encoder = self.create_encoder()
        if self.flag_AE == 'MD-CNN-AE':
            for i in range(self.N_z):
                setattr(self, 'decoder' + str(i + 1), self.create_decoder())
        else:
            self.decoder = self.create_decoder()

    def create_encoder(self):

        N_y_sub = self.N_y
        N_x_sub = self.N_x
        i_seq = 0

        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.layers.Input(shape=(self.N_x, self.N_y, self.K)))

        # 1st: conv & max pooling (if m x n = 384 x 192)
        if N_x_sub == 384:
            encoder.add(tf.keras.layers.Conv2D(self.filter_seq_e[i_seq], self.k_size, activation=self.activation, padding='same', kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
            encoder.add(tf.keras.layers.MaxPooling2D(pool_size=self.p_size, padding=self.p_pad, strides=self.p_strides))
            N_x_sub = (N_x_sub) // self.p_strides
            N_y_sub = (N_y_sub) // self.p_strides
        i_seq += 1

        # 2nd: conv & max pooling
        if N_x_sub == 192:
            encoder.add(layers.Conv2D(self.filter_seq_e[i_seq], self.k_size, activation=self.activation, padding='same', kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
            encoder.add(layers.MaxPooling2D(pool_size=self.p_size, padding=self.p_pad, strides=self.p_strides))
            N_x_sub = (N_x_sub) // self.p_strides
            N_y_sub = (N_y_sub) // self.p_strides
        i_seq += 1

        # 3rd: conv & max pooling
        encoder.add(layers.Conv2D(self.filter_seq_e[i_seq], self.k_size, activation=self.activation, padding='same', kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
        encoder.add(layers.MaxPooling2D(pool_size=self.p_size, padding=self.p_pad, strides=self.p_strides))
        N_x_sub = (N_x_sub) // self.p_strides
        N_y_sub = (N_y_sub) // self.p_strides
        i_seq += 1

        # 4th: conv & max pooling
        encoder.add(layers.Conv2D(self.filter_seq_e[i_seq], self.k_size, activation=self.activation, padding='same', kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
        encoder.add(layers.MaxPooling2D(pool_size=self.p_size, padding=self.p_pad, strides=self.p_strides))
        N_x_sub = (N_x_sub) // self.p_strides
        N_y_sub = (N_y_sub) // self.p_strides
        i_seq += 1

        # 5th: conv & max pooling
        encoder.add(layers.Conv2D(self.filter_seq_e[i_seq], self.k_size, activation=self.activation, padding='same', kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
        encoder.add(layers.MaxPooling2D(pool_size=self.p_size, padding=self.p_pad, strides=self.p_strides))
        N_x_sub = (N_x_sub) // self.p_strides
        N_y_sub = (N_y_sub) // self.p_strides
        i_seq += 1

        # 6th: conv & max pooling
        encoder.add(layers.Conv2D(self.filter_seq_e[i_seq], self.k_size, activation=self.activation, padding='same', kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
        encoder.add(layers.MaxPooling2D(pool_size=self.p_size, padding=self.p_pad, strides=self.p_strides))
        N_x_sub = (N_x_sub) // self.p_strides
        N_y_sub = (N_y_sub) // self.p_strides

        # fully connected
        encoder.add(layers.Reshape([N_y_sub * N_x_sub * self.filter_seq_e[i_seq]]))
        if self.drp_rate > 0:
            encoder.add(layers.Dropout(self.drp_rate))
        i_seq += 1
        if self.flag_struct != 'simple':
            encoder.add(layers.Dense(self.filter_seq_e[i_seq], activation=self.activation, kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))

        encoder.add(layers.Dense(self.N_e, activation=self.activation, kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))

        self.N_y_sub = N_y_sub
        self.N_x_sub = N_x_sub

        return encoder
    def create_decoder(self):

        N_y_sub = self.N_y_sub
        N_x_sub = self.N_x_sub
        i_seq = 0

        decoder = tf.keras.Sequential()

        # fully connected
        if self.flag_struct != 'simple':
            decoder.add(layers.Dense(self.filter_seq_d[i_seq], activation=self.activation, kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
            i_seq += 1

        decoder.add(layers.Dense(N_y_sub * N_x_sub * self.filter_seq_d[i_seq], activation=self.activation, kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
        if self.drp_rate > 0:
            decoder.add(layers.Dropout(self.drp_rate))
        decoder.add(layers.Reshape([N_x_sub, N_y_sub, self.filter_seq_d[i_seq]]))
        i_seq += 1

        # 1st: upsampling & (7th) conv
        decoder.add(layers.UpSampling2D(self.p_size))
        decoder.add(layers.Conv2D(self.filter_seq_d[i_seq], self.k_size, activation=self.activation, padding='same', kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
        i_seq += 1

        # 2nd: upsampling & (8th) conv
        decoder.add(layers.UpSampling2D(self.p_size))
        decoder.add(layers.Conv2D(self.filter_seq_d[i_seq], self.k_size, activation=self.activation, padding='same', kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
        i_seq += 1

        # 3rd: upsampling & (9th) conv
        decoder.add(layers.UpSampling2D(self.p_size))
        decoder.add(layers.Conv2D(self.filter_seq_d[i_seq], self.k_size, activation=self.activation, padding='same', kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
        i_seq += 1

        # 4th: upsampling & (10th) conv
        decoder.add(layers.UpSampling2D(self.p_size))
        decoder.add(layers.Conv2D(self.filter_seq_d[i_seq], self.k_size, activation=self.activation, padding='same', kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
        i_seq += 1

        # 5th: upsampling & (11th) conv
        if (self.flag_struct != 'simple') and (self.N_x == 192):
            decoder.add(layers.UpSampling2D(self.p_size))
            decoder.add(layers.Conv2D(self.filter_seq_d[i_seq], self.k_size, activation=self.activation, padding='same', kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))
            i_seq += 1

        # 6th: upsampling & (12th) conv  (if m x n = 392 x 196)
        if self.N_x == 384:
            decoder.add(layers.UpSampling2D(self.p_size))
            decoder.add(layers.Conv2D(self.filter_seq_d[i_seq], self.k_size, activation=self.activation, padding='same', kernel_regularizer=l2(self.k_reg), bias_regularizer=l2(self.k_b)))

        if self.flag_struct == 'simple':
            decoder.add(layers.UpSampling2D(self.p_size))

        decoder.add(layers.Conv2D(self.K, self.k_size, activation=self.activation, padding='same'))

        return decoder


class MD_CNN_AE(AE, Model):
    """
    Modal CNN-AE (linear summation of modes)
    """

    def __init__(self, PARAMS, FLAGS):

        # Call parent constructor
        super(MD_CNN_AE, self).__init__(PARAMS=PARAMS, FLAGS=FLAGS)


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
        for i in range(self.N_z):
            lat_vector = tf.keras.layers.Lambda(lambda x: x[:, i:i + 1])(encoded)
            decoded_sub = getattr(self, 'decoder'+str(i+1))(lat_vector)

            if i == 0:
                decoded = tf.keras.layers.Add()([decoded_sub])
            else:
                decoded = tf.keras.layers.Add()([decoded, decoded_sub])

        return decoded

class CNN_HAE(AE, Model):
    """
    Hierarchical convolutional AE
    """

    def __init__(self, PARAMS, FLAGS):

        # Call parent constructor
        super(CNN_HAE, self).__init__(PARAMS=PARAMS, FLAGS=FLAGS)

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
    """
    Variational convolutional AE
    """

    def __init__(self, PARAMS, FLAGS):

        # Call parent constructor
        super(CNN_VAE, self).__init__(PARAMS=PARAMS, FLAGS=FLAGS)


    def get_latent_vector(self, input_img):

        encoded = self.encoder(input_img)
        z_mean, z_logvar = tf.split(encoded, num_or_size_splits=2, axis=1)
        # z = self.sampling(z_mean, z_logvar)
        return z_mean
    def sampling(self, z_mean, z_log_sigma):

        epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], self.N_z),
                                  mean=0., stddev=1.0)
        return z_mean + tf.keras.backend.exp(z_log_sigma) * epsilon

    def loss_fn(self, input_img, decoded, z_logvar, z_mean):
        rec_loss = tf.keras.losses.mse(tf.keras.backend.reshape(input_img, (-1,)), tf.keras.backend.reshape(decoded, (-1,)))
        kl_loss = 1 + z_logvar - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_logvar)
        kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.keras.backend.mean(rec_loss + self.beta * kl_loss)
        return vae_loss

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

        vae_loss = self.loss_fn(input_img, decoded, z_logvar, z_mean)
        self.add_loss(vae_loss)
        #self.add_metric(rec_loss)
        #self.add_metric(kl_loss)

        return decoded

class C_CNN_AE(AE, Model):
    """
    Basic convolutional AE
    """

    def __init__(self, PARAMS, FLAGS):

        # Call parent constructor
        super(C_CNN_AE, self).__init__(PARAMS=PARAMS, FLAGS=FLAGS)

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
