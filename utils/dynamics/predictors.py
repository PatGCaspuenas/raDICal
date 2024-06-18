# PACKAGES
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.regularizers import l2
import tensorflow as tf

class LSTM(Model):

    def __init__(self, params, flags):

        super(LSTM, self).__init__()

        self.nr = params['AE']['nr']
        self.nc = params['flow']['nc']

        self.n_input = self.nr + self.nc
        self.n_output = self.nr

        self.np = params['dyn']['np']
        self.d = params['dyn']['d']

        self.lstm_units = params['LSTM']['lstm_units']
        self.dense_units = params['LSTM']['dense_units']
        self.dropout = params['LSTM']['dropout']

        self.flag_control = flags['dyn']['control']

        self.lstm = self.create_lstm_layers()
        self.predictor = self.create_dense_layers()

    def create_lstm_layers(self):

        lstm =  tf.keras.Sequential()
        lstm.add(tf.keras.layers.Input(shape=(self.d, self.n_input)))

        for units in self.lstm_units:
            if units == self.lstm_units[-1]:
                lstm.add(layers.LSTM(units, return_sequences=False))
            else:
                lstm.add(layers.LSTM(units, return_sequences=True))

        for units, drop in zip(self.dense_units, self.dropout):
            lstm.add(layers.Dense(units, activation='tanh'))
            if drop > 0:
                lstm.add(layers.Dropout(self.drop))

        return lstm

    def create_dense_layers(self):

        predictor = tf.keras.Sequential()

        predictor.add(layers.Dense(self.np * self.n_output, activation='tanh'))
        predictor.add(layers.Reshape([self.np, self.n_output]))

        return predictor

    def call(self, input):

        if not self.flag_control:
            output = self.predictor(self.lstm(input))
        else:
            input_feature = input[0]
            control_vector = input[1]

            control_vector = np.reshape(control_vector,[self.nc * self.np])

            encoded = self.lstm(input_feature)
            output = self.predictor(layers.Concatenate(axis=1)([encoded, control_vector]))

        return output

class NARX(Model):

    def __init__(self, params, flags):

        super(NARX, self).__init__()

        self.nr = params['AE']['nr']
        self.nc = params['flow']['nc']

        self.np = params['dyn']['np']
        self.d_c = params['dyn']['d_c']
        self.d_lat = params['dyn']['d_lat']

        self.units = params['NARX']['units']
        self.dropout = params['LSTM']['dropout']

        self.flag_control = flags['dyn']['control']

        self.MLP = self.create_MLP_cell()

    def create_MLP_cell(self):

        MLP = tf.keras.Sequential()
        MLP.add(tf.keras.layers.Input(shape=((self.d_c + 1) * self.nc + self.d_lat * self.nr)))

        for units, drop in zip(self.units, self.dropout):
            MLP.add(layers.Dense(units, activation='tanh'))
            if drop > 0:
                MLP.add(layers.Dropout(self.drop))

        MLP.add(layers.Dense(self.nr, activation='linear'))
        MLP.add(layers.Reshape([1, self.nr]))

        return MLP

    def call(self, input):

        output = []

        if self.flag_control:

            input_feature = np.reshape(input[0], (None, self.nr * self.d_lat)) # Same order as tf
            input_control = np.reshape(input[1], (None, self.nc * self.d_c))
            output_control = np.reshape(input[2][None, 0, :], (None, self.nc))

            input_t = np.concatenate((input_feature, input_control, output_control), axis=1)
        else:

            input_t = np.reshape(input, (None, self.nr * self.d_lat))

        output_t = self.MLP(input_t)
        output.append(output_t)

        for n in range(1, self.np):

            output_t = np.reshape(output_t, (None, self.nr))

            if self.flag_control:

                input_feature = np.concatenate((input_feature[None, self.nr:], output_t), axis=1)
                input_control = np.concatenate((input_control[None, self.nc:], output_control), axis=1)
                output_control = np.reshape(input[2][:, n, :], (None, self.nc))

                input_t = np.concatenate((input_feature, input_control, output_control), axis=1)
            else:

                input_t = np.concatenate((input_t[None, self.nr:], output_t), axis=1)

            output_t = self.MLP(input_t)
            output.append(output_t)

        # output.shape => (time, batch, features)
        output = tf.stack(output)
        # output.shape => (batch, time, features)
        output = tf.transpose(output, [1, 0, 2])
        return output

