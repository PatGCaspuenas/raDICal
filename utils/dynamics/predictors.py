# PACKAGES
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.regularizers import l2
import tensorflow as tf
import random

class LSTM(Model):

    def __init__(self, params, flags):

        super(LSTM, self).__init__()

        self.nr = params['AE']['nr']
        self.nc = params['flow']['nc']

        self.n_input = self.nr + self.nc
        self.n_output = self.nr

        self.np = params['dyn']['np']
        self.d = params['dyn']['d']
        self.act = params['dyn']['act']

        self.dt = params['dyn']['dt']
        self.dt_lat = params['dyn']['dt_lat']
        self.nTDL = self.d  * int( self.dt_lat / self.dt ) -  int( self.dt_lat / self.dt  - 1)

        self.k_reg = params['dyn']['kreg']

        self.lstm_units = params['LSTM']['lstm_units']
        self.dense_units = params['LSTM']['dense_units']
        self.dropout = params['LSTM']['dropout']

        self.flag_control = flags['dyn']['control']

        self.lstm = self.create_lstm_layers()
        self.predictor = self.create_dense_layers()

    def create_lstm_layers(self):

        lstm = tf.keras.Sequential()
        lstm.add(tf.keras.layers.Input(shape=(self.d, self.n_input)))

        count = 0
        for units in self.lstm_units:
            if count == (len(self.lstm_units) - 1):
                if count == 0:
                    lstm.add(layers.LSTM(units, return_sequences=False, kernel_regularizer=l2(self.k_reg)))
                else:
                    lstm.add(layers.LSTM(units, return_sequences=False))
            else:
                if count == 0:
                    lstm.add(layers.LSTM(units, return_sequences=True, kernel_regularizer=l2(self.k_reg)))
                else:
                    lstm.add(layers.LSTM(units, return_sequences=True))
                count += 1

        for units, drop in zip(self.dense_units, self.dropout):
            lstm.add(layers.Dense(units, activation=self.act))
            if drop > 0:
                lstm.add(layers.Dropout(drop))

        return lstm

    def create_dense_layers(self):

        predictor = tf.keras.Sequential()

        predictor.add(layers.Dense(self.np * self.n_output, activation=self.act))
        predictor.add(layers.Reshape([self.np, self.n_output]))

        return predictor

    def predict(self, input, nt):

        if self.flag_control:
            n_batch = np.shape(input[0])[0]
        else:
            n_batch = np.shape(input)[0]

        n_steps = int(np.ceil(nt / self.np))

        output = np.zeros((n_batch, nt, self.n_output))

        if self.flag_control:
            input_state = input[0]
            input_control = input[1]
            output_control = input[2]

            input_feature = np.concatenate((input_state, input_control), axis=2)
        else:
            input_feature = input

        for t in range(n_steps):

            if not self.flag_control:

                if self.dt != self.dt_lat:
                    indices = np.sort(random.sample(range(self.nTDL), self.d))
                    output_step = self.predictor(self.lstm(tf.gather(input, indices=indices, axis=1)))
                else:
                    output_step = self.predictor(self.lstm(input_feature))

                prediction_step = np.concatenate((input_feature, output_step), axis=1)
            else:
                control_step = np.reshape(output_control[:, (self.np * t):(self.np * (t+1)), :],(n_batch, self.nc * self.np))

                if self.dt != self.dt_lat:
                    indices = np.sort(random.sample(range(self.nTDL), self.d))
                    encoded = self.lstm(tf.gather(input_feature, indices=indices, axis=1))
                else:
                    encoded = self.lstm(input_feature)

                output_step = self.predictor(layers.Concatenate(axis=1)([encoded, control_step]))

                prediction_step = np.concatenate((input_feature, np.concatenate((output_step, output_control[:, (self.np * t):(self.np * (t+1)), :]), axis=2)), axis=1)

            if (t == (n_steps - 1)) & ((nt / self.np) < n_steps):
                nlast = nt - int(np.floor(nt/self.np)) * self.np
                output[:, (self.np * t):, :] = output_step[:, :nlast, :]
            else:
                output[:, (self.np * t):(self.np * (t + 1)), :] = output_step[:, :, :]

            input_feature = prediction_step[:, -self.nTDL:, :]

        return output

    def call(self, input):

        if not self.flag_control:
            if self.dt != self.dt_lat:
                indices = np.sort(random.sample(range(self.nTDL), self.d))
                output = self.predictor(self.lstm(tf.gather(input, indices=indices, axis=1)))
            else:
                output = self.predictor(self.lstm(input))
        else:
            input_state = input[0]
            input_control = input[1]
            output_control = input[2]

            input_feature = tf.concat([input_state, input_control], axis=2)

            output_control = tf.keras.layers.Reshape((self.nc * self.np,))(output_control)

            if self.dt != self.dt_lat:
                indices = np.sort(random.sample(range(self.nTDL), self.d))
                encoded = self.lstm(tf.gather(input_feature, indices=indices, axis=1))
            else:
                encoded = self.lstm(input_feature)

            output = self.predictor(layers.Concatenate(axis=1)([encoded, output_control]))

        return output

class NARX(Model):

    def __init__(self, params, flags):

        super(NARX, self).__init__()

        self.nr = params['AE']['nr']
        self.nc = params['flow']['nc']

        self.np = params['dyn']['np']
        self.d = params['dyn']['d']
        self.act = params['dyn']['act']

        self.k_reg = params['dyn']['kreg']

        self.dt = params['dyn']['dt']
        self.dt_lat = params['dyn']['dt_lat']
        self.irreg = params['dyn']['irreg']
        self.nTDL = self.d  * int( self.dt_lat / self.dt ) -  int( self.dt_lat / self.dt  - 1)

        self.units = params['NARX']['units']
        self.dropout = params['NARX']['dropout']

        self.flag_control = flags['dyn']['control']
        self.flag_multi_train = flags['dyn']['multi_train']

        self.MLP = self.create_MLP_cell()

    def create_MLP_cell(self):

        MLP = tf.keras.Sequential()
        if self.flag_multi_train:
            MLP.add(tf.keras.layers.Input(shape=((self.d + self.np) * self.nc + self.d * self.nr)))
        else:
            MLP.add(tf.keras.layers.Input(shape=((self.d + 1) * self.nc + self.d * self.nr)))

        count = 0
        for units, drop in zip(self.units, self.dropout):
            if count == 0:
                MLP.add(layers.Dense(units, activation=self.act, kernel_regularizer=l2(self.k_reg)))
            else:
                MLP.add(layers.Dense(units, activation=self.act))
            if drop > 0:
                MLP.add(layers.Dropout(drop))

        if self.flag_multi_train:
            MLP.add(layers.Dense(self.nr * self.np, activation='linear'))
            MLP.add(layers.Reshape([self.np, self.nr]))
        else:
            MLP.add(layers.Dense(self.nr, activation='linear'))
            MLP.add(layers.Reshape([1, self.nr]))

        return MLP

    def predict(self, input, nt):

        if self.flag_control:
            n_batch = np.shape(input[0])[0]
        else:
            n_batch = np.shape(input)[0]

        output = np.zeros((n_batch, nt, self.nr))

        if self.flag_control:

            TDL = np.concatenate((input[0], input[1]), axis=2)
            if self.dt != self.dt_lat:
                if not self.irreg:
                    indices = np.arange(0, self.nTDL, self.d)
                else:
                    indices = np.sort(random.sample(range(self.nTDL), self.d))
                input_feature = tf.keras.layers.Reshape((self.nr * self.d,))(
                    tf.gather(TDL, indices=indices, axis=1)[:, :, :self.nr])
                input_control = tf.keras.layers.Reshape((self.nc * self.d,))(
                    tf.gather(TDL, indices=indices, axis=1)[:, :, self.nr:])
            else:
                input_feature = tf.keras.layers.Reshape((self.nr * self.d,))(input[0])
                input_control = tf.keras.layers.Reshape((self.nc * self.d,))(input[1])

            if self.flag_multi_train:
                output_control = tf.keras.layers.Reshape((self.nc * self.np,))(input[2][:, :self.np, :])
            else:
                output_control = tf.keras.layers.Reshape((self.nc,))(input[2][:, 0, :])

            input_t = np.concatenate((input_feature, input_control, output_control), axis=1)
        else:

            TDL = tf.identity(input)
            if self.dt != self.dt_lat:
                if not self.irreg:
                    indices = np.arange(0, self.nTDL, self.d)
                else:
                    indices = np.sort(random.sample(range(self.nTDL), self.d))
                input_t = tf.keras.layers.Reshape((self.nr * self.d,))(tf.gather(TDL, indices=indices, axis=1))
            else:
                input_t = tf.keras.layers.Reshape((self.nr * self.d,))(TDL)

            # input_t = np.reshape(input, (1, self.nr * self.d))

        output_t = self.MLP(input_t)
        if self.flag_multi_train:
            output[:, :self.np, :] = output_t[:, :self.np, :]
        else:
            output[:, 0, :] = output_t[:, 0, :]

        if not self.flag_multi_train:
            for n in range(1, nt):

                # output_t = np.reshape(output_t, (1, self.nr))

                if self.flag_control:

                    TDL = np.concatenate((TDL[:, 1:, :], np.concatenate((output_t, input[2][:, n - 1:n, :]), axis=2)),
                                         axis=1)
                    if self.dt != self.dt_lat:
                        if not self.irreg:
                            indices = np.arange(0, self.nTDL, self.d)
                        else:
                            indices = np.sort(random.sample(range(self.nTDL), self.d))
                        input_feature = tf.keras.layers.Reshape((self.nr * self.d,))(
                            tf.gather(TDL, indices=indices, axis=1)[:, :, :self.nr])
                        input_control = tf.keras.layers.Reshape((self.nc * self.d,))(
                            tf.gather(TDL, indices=indices, axis=1)[:, :, self.nr:])
                    else:
                        input_feature = tf.keras.layers.Reshape((self.nr * self.d,))(TDL[:, :, :self.nr])
                        input_control = tf.keras.layers.Reshape((self.nc * self.d,))(TDL[:, :, self.nr:])

                    # input_feature = tf.concat([input_feature[:, self.nr:], output_t], 1)
                    # input_control = tf.concat([input_control[:, self.nc:], output_control], 1)

                    output_control = tf.keras.layers.Reshape((self.nc,))(input[2][:, n, :])

                    input_t = tf.concat([input_feature, input_control, output_control], 1)

                else:
                    TDL = np.concatenate((TDL[:, 1:, :], output_t), axis=1)
                    if self.dt != self.dt_lat:
                        if not self.irreg:
                            indices = np.arange(0, self.nTDL, self.d)
                        else:
                            indices = np.sort(random.sample(range(self.nTDL), self.d))
                        input_t = tf.keras.layers.Reshape((self.nr * self.d,))(tf.gather(TDL, indices=indices, axis=1))
                    else:
                        input_t = tf.keras.layers.Reshape((self.nr * self.d,))(TDL)
                    # input_t = np.concatenate((input_t[:, self.nr:], output_t), axis=1)

                output_t = self.MLP(input_t)
                output[:, n, :] = output_t[:, 0, :]
        else:
            n_steps = int(np.ceil(nt / self.np))

            for t in range(1, n_steps):

                if self.flag_control:
                    TDL = np.concatenate((TDL[:, self.np:, :],
                                          np.concatenate((output_t, input[2][:, (self.np * (t - 1)):(self.np * t), :]),
                                                         axis=2)),
                                         axis=1)
                    if self.dt != self.dt_lat:
                        if not self.irreg:
                            indices = np.arange(0, self.nTDL, self.d)
                        else:
                            indices = np.sort(random.sample(range(self.nTDL), self.d))
                        input_feature = tf.keras.layers.Reshape((self.nr * self.d,))(
                            tf.gather(TDL, indices=indices, axis=1)[:, :, :self.nr])
                        input_control = tf.keras.layers.Reshape((self.nc * self.d,))(
                            tf.gather(TDL, indices=indices, axis=1)[:, :, self.nr:])
                    else:

                        input_feature = tf.keras.layers.Reshape((self.nr * self.d,))(TDL[:, :, :self.nr])
                        input_control = tf.keras.layers.Reshape((self.nc * self.d,))(TDL[:, :, self.nr:])

                    # input_feature = tf.concat([input_feature[:, self.nr:], output_t], 1)
                    # input_control = tf.concat([input_control[:, self.nc:], output_control], 1)

                    if (t == (n_steps - 1)) & ((nt / self.np) < n_steps):
                        nlast = nt - int(np.floor(nt / self.np)) * self.np
                        output_control = tf.keras.layers.Reshape((self.nc * self.np,))(np.concatenate(
                            (input[2][:, (self.np * t):, :], input[2][:, -1:, :].repeat(self.np - nlast, axis=1)),
                            axis=1))
                    else:
                        output_control = tf.keras.layers.Reshape((self.nc * self.np,))(
                            input[2][:, (self.np * t):(self.np * (t + 1)), :])

                    input_t = tf.concat([input_feature, input_control, output_control], 1)

                else:

                    TDL = np.concatenate((TDL[:, self.np:, :], output_t), axis=1)
                    if self.dt != self.dt_lat:
                        if not self.irreg:
                            indices = np.arange(0, self.nTDL, self.d)
                        else:
                            indices = np.sort(random.sample(range(self.nTDL), self.d))
                        input_t = tf.keras.layers.Reshape((self.nr * self.d,))(tf.gather(TDL, indices=indices, axis=1))
                    else:
                        input_t = tf.keras.layers.Reshape((self.nr * self.d,))(TDL)
                    # input_t = np.concatenate((input_t[:, self.nr:], output_t), axis=1)

                output_t = self.MLP(input_t)

                if (t == (n_steps - 1)) & ((nt / self.np) < n_steps):
                    nlast = nt - int(np.floor(nt / self.np)) * self.np
                    output[:, (self.np * t):, :] = output_t[:, :nlast, :]
                else:
                    output[:, (self.np * t):(self.np * (t + 1)), :] = output_t[:, :self.np, :]

        return output

    def call(self, input):

        if self.flag_control:

            TDL = tf.concat([input[0], input[1]], 2)
            if self.dt != self.dt_lat:
                if not self.irreg:
                    indices = np.arange(0, self.nTDL, self.d)
                else:
                    indices = np.sort(random.sample(range(self.nTDL), self.d))
                input_feature = tf.keras.layers.Reshape((self.nr * self.d,))(
                    tf.gather(TDL, indices=indices, axis=1)[:, :, :self.nr])
                input_control = tf.keras.layers.Reshape((self.nc * self.d,))(
                    tf.gather(TDL, indices=indices, axis=1)[:, :, self.nr:])
            else:
                input_feature = tf.keras.layers.Reshape((self.nr * self.d,))(input[0])
                input_control = tf.keras.layers.Reshape((self.nc * self.d,))(input[1])

            if self.flag_multi_train:
                output_control = tf.keras.layers.Reshape((self.nc * self.np,))(input[2][:, :self.np, :])
            else:
                output_control = tf.keras.layers.Reshape((self.nc,))(input[2][:, 0, :])

            input_t = tf.concat([input_feature, input_control, output_control], 1)
        else:

            TDL = tf.identity(input)
            if self.dt != self.dt_lat:
                if not self.irreg:
                    indices = np.arange(0, self.nTDL, self.d)
                else:
                    indices = np.sort(random.sample(range(self.nTDL), self.d))
                input_t = tf.keras.layers.Reshape((self.nr * self.d,))(tf.gather(TDL, indices=indices, axis=1))
            else:
                input_t = tf.keras.layers.Reshape((self.nr * self.d,))(TDL)

        output_t = self.MLP(input_t)
        output = tf.identity(output_t)

        if not self.flag_multi_train:
            for n in range(1, self.np):

                # output_t = tf.keras.layers.Reshape((self.nr,))(output_t)

                if self.flag_control:

                    TDL = tf.concat([TDL[:, 1:, :], tf.concat([output_t, input[2][:, n - 1:n, :]], 2)], 1)
                    if self.dt != self.dt_lat:
                        if not self.irreg:
                            indices = np.arange(0, self.nTDL, self.d)
                        else:
                            indices = np.sort(random.sample(range(self.nTDL), self.d))
                        input_feature = tf.keras.layers.Reshape((self.nr * self.d,))(
                            tf.gather(TDL, indices=indices, axis=1)[:, :, :self.nr])
                        input_control = tf.keras.layers.Reshape((self.nc * self.d,))(
                            tf.gather(TDL, indices=indices, axis=1)[:, :, self.nr:])
                    else:
                        input_feature = tf.keras.layers.Reshape((self.nr * self.d,))(TDL[:, :, :self.nr])
                        input_control = tf.keras.layers.Reshape((self.nc * self.d,))(TDL[:, :, self.nr:])

                    # input_feature = tf.concat([input_feature[:, self.nr:], output_t], 1)
                    # input_control = tf.concat([input_control[:, self.nc:], output_control], 1)

                    output_control = tf.keras.layers.Reshape((self.nc,))(input[2][:, n, :])

                    input_t = tf.concat([input_feature, input_control, output_control], 1)
                else:
                    TDL = tf.concat([TDL[:, 1:, :], output_t], 1)
                    if self.dt != self.dt_lat:
                        if not self.irreg:
                            indices = np.arange(0, self.nTDL, self.d)
                        else:
                            indices = np.sort(random.sample(range(self.nTDL), self.d))
                        input_t = tf.keras.layers.Reshape((self.nr * self.d,))(tf.gather(TDL, indices=indices, axis=1))
                    else:
                        input_t = tf.keras.layers.Reshape((self.nr * self.d,))(TDL)
                    # input_t = tf.concat([input_t[:, self.nr:], output_t], 1)

                output_t = self.MLP(input_t)
                output = tf.concat([output, output_t], 1)

        return output