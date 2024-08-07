# PACKAGES
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras.regularizers import l2
import tensorflow as tf

# LOCAL FUNCTIONS
from utils.dynamics.optimizers import LM_Trainer
from utils.dynamics.config import cond_lm

class DYN(object):
    """
    Dynamical predictor parent class, containing initialization of parameters, NN structure and common utils
    """

    def __init__(self, PARAMS, FLAGS):

        super(DYN, self).__init__()

        self.flag_dyn = FLAGS['DYN']['type']            # Type of dynamical predictor (LSTM, NARX)
        self.flag_control = FLAGS['DYN']['control']     # 1 if control embedded in NN, 0 otherwise
        self.flag_recurrent = FLAGS['DYN']['recurrent'] # 1 if NARX follows a recurrent training pattern, 0 otherwise
        self.flag_optimizer = FLAGS['DYN']['optimizer'] # Type of optimzer (LM, Adam)

        self.N_z = PARAMS['AE']['N_z']                  # Number of latent coordinates
        self.N_c = PARAMS['flow']['N_c']                # Number of control coordinates

        self.N_input = self.N_z + self.N_c
        self.N_output = self.N_z

        self.w_p = PARAMS['DYN']['w_p']                 # Number of trained prediction time instants
        self.w_d = PARAMS['DYN']['w_d']                 # Number of delay time instants
        self.activation = PARAMS['DYN']['activation']   # Type of activation function

        self.DT = PARAMS['DYN']['DT']                   # Delta time

        self.k_reg = PARAMS['DYN']['k_reg']             # l2 regularization of dense layers
        self.drp_rate = PARAMS['DYN']['drp_rate']       # dropout rate of dense layers

        self.hidden_units = PARAMS['DYN']['hidden_units'] # Number of units in LSTM cells (only LSTM)
        self.dense_units = PARAMS['DYN']['dense_units']   # Number of units in dense layers (LSTM & NARX)

        # Create NNs depending on type
        if self.flag_dyn == 'LSTM':
            self.LSTMs = self.LSTM_layers()
            self.MLPs = self.MLP_layers()
        elif self.flag_dyn == 'NARX':
            self.MLP = self.MLP_cell()

        # Only for LEVENBERG-MARQUADT OPTIMIZER
        if self.flag_optimizer == 'LM':
            # Used to backup and restore model variables.
            self._backup_variables = []

            # Since training updates are computed with shape (num_variables, 1),
            # self._splits and self._shapes are needed to split and reshape the
            # updates so that they can be applied to the model trainable_variables.
            self._splits = []
            self._shapes = []

            for variable in self.trainable_variables:
                variable_shape = tf.shape(variable)
                variable_size = tf.reduce_prod(variable_shape)
                backup_variable = tf.Variable(
                    tf.zeros_like(variable),
                    trainable=False)

                self._backup_variables.append(backup_variable)
                self._splits.append(variable_size)
                self._shapes.append(variable_shape)

            self._num_variables = tf.reduce_sum(self._splits).numpy().item()
            self._num_outputs = None

    def MLP_cell(self):

        MLP = tf.keras.Sequential()
        if not self.flag_recurrent:
            MLP.add(tf.keras.layers.Input(shape=((self.w_d + self.w_p) * self.N_c + self.w_d * self.N_z)))
        else:
            MLP.add(tf.keras.layers.Input(shape=((self.w_d + 1) * self.N_c + self.w_d * self.N_z)))

        count = 0
        for units, drop in zip(self.dense_units, self.drp_rate):
            if count == 0:
                MLP.add(layers.Dense(units, activation=self.activation, kernel_regularizer=l2(self.k_reg)))
            else:
                MLP.add(layers.Dense(units, activation=self.activation))
            if drop > 0:
                MLP.add(layers.Dropout(drop))

        if not self.flag_recurrent:
            MLP.add(layers.Dense(self.N_z * self.w_p, activation='linear'))
            MLP.add(layers.Reshape([self.w_p, self.N_z]))
        else:
            MLP.add(layers.Dense(self.N_z, activation='linear'))
            MLP.add(layers.Reshape([1, self.N_z]))

        return MLP

    def LSTM_layers(self):

        LSTMs = tf.keras.Sequential()
        LSTMs.add(tf.keras.layers.Input(shape=(self.w_d, self.N_input)))

        count = 0
        for units in self.hidden_units:
            if count == (len(self.hidden_units) - 1):
                if count == 0:
                    LSTMs.add(layers.LSTM(units, return_sequences=False, kernel_regularizer=l2(self.k_reg)))
                else:
                    LSTMs.add(layers.LSTM(units, return_sequences=False))
            else:
                if count == 0:
                    LSTMs.add(layers.LSTM(units, return_sequences=True, kernel_regularizer=l2(self.k_reg)))
                else:
                    LSTMs.add(layers.LSTM(units, return_sequences=True))
                count += 1
        LSTMs.add(layers.Flatten())
        for units, drop in zip(self.dense_units, self.drp_rate):
            LSTMs.add(layers.Dense(units, activation=self.activation))
            if drop > 0:
                LSTMs.add(layers.Dropout(drop))

        return LSTMs
    def MLP_layers(self):

        MLPs = tf.keras.Sequential()

        MLPs.add(layers.Dense(self.w_p * self.N_output, activation='linear'))
        MLPs.add(layers.Reshape([self.w_p, self.N_output]))

        return MLPs

class LSTM(DYN, LM_Trainer if cond_lm else Model):
    """
    LSTM method
    """

    def __init__(self, PARAMS, FLAGS):

        super(LSTM, self).__init__(PARAMS=PARAMS, FLAGS=FLAGS)

    def predict(self, input, w_prop):

        if self.flag_control:
            n_batch = np.shape(input[0])[0]
        else:
            n_batch = np.shape(input)[0]

        N_steps = int(np.ceil(w_prop / self.w_p))

        output = np.zeros((n_batch, w_prop, self.N_output))

        if self.flag_control:
            input_state = input[0]
            input_control = input[1]
            output_control = input[2]

            input_feature = np.concatenate((input_state, input_control), axis=2)
        else:
            input_feature = input

        for t in range(N_steps):

            if not self.flag_control:

                output_step = self.MLPs(self.LSTMs(input_feature))

                prediction_step = np.concatenate((input_feature, output_step), axis=1)
            else:
                control_step = np.reshape(output_control[:, (self.w_p * t):(self.w_p * (t+1)), :],(n_batch, self.N_c * self.w_p))

                encoded = self.LSTMs(input_feature)

                output_step = self.MLPs(layers.Concatenate(axis=1)([encoded, control_step]))

                prediction_step = np.concatenate((input_feature, np.concatenate((output_step, output_control[:, (self.w_p * t):(self.w_p * (t+1)), :]), axis=2)), axis=1)

            if (t == (N_steps - 1)) & ((w_prop / self.w_p) < N_steps):
                N_last = w_prop - int(np.floor(w_prop/self.w_p)) * self.w_p
                output[:, (self.w_p * t):, :] = output_step[:, :N_last, :]
            else:
                output[:, (self.w_p * t):(self.w_p * (t + 1)), :] = output_step[:, :, :]

            input_feature = prediction_step[:, -self.w_d:, :]

        return output

    def call(self, input):

        if not self.flag_control:
            output = self.MLPs(self.LSTMs(input))
        else:
            input_state = input[0]
            input_control = input[1]
            output_control = input[2]

            input_feature = tf.concat([input_state, input_control], axis=2)

            output_control = tf.keras.layers.Reshape((self.N_c * self.w_p,))(output_control)

            encoded = self.LSTMs(input_feature)

            output = self.MLPs(layers.Concatenate(axis=1)([encoded, output_control]))

        return output

class NARX(DYN, LM_Trainer if cond_lm else Model):
    """
    NARX method
    """

    def __init__(self, PARAMS, FLAGS):

        super(NARX, self).__init__(PARAMS=PARAMS, FLAGS=FLAGS)

    def predict(self, input, w_prop):

        if self.flag_control:
            n_batch = np.shape(input[0])[0]
        else:
            n_batch = np.shape(input)[0]

        output = np.zeros((n_batch, w_prop, self.N_z))

        if self.flag_control:

            TDL = np.concatenate((input[0], input[1]), axis=2)
            input_feature = tf.keras.layers.Reshape((self.N_z * self.w_d,))(input[0])
            input_control = tf.keras.layers.Reshape((self.N_c * self.w_d,))(input[1])

            if not self.flag_recurrent:
                output_control = tf.keras.layers.Reshape((self.N_c * self.w_p,))(input[2][:, :self.w_p, :])
            else:
                output_control = tf.keras.layers.Reshape((self.N_c,))(input[2][:, 0, :])

            input_t = np.concatenate((input_feature, input_control, output_control), axis=1)
        else:

            TDL = tf.identity(input)
            input_t = tf.keras.layers.Reshape((self.N_z * self.w_d,))(TDL)

        output_t = self.MLP(input_t)
        if not self.flag_recurrent:
            output[:, :self.w_p, :] = output_t[:, :self.w_p, :]
        else:
            output[:, 0, :] = output_t[:, 0, :]

        if self.flag_recurrent:
            for n in range(1, w_prop):

                if self.flag_control:

                    TDL = np.concatenate((TDL[:, 1:, :], np.concatenate((output_t, input[2][:, n - 1:n, :]), axis=2)),axis=1)
                    input_feature = tf.keras.layers.Reshape((self.N_z * self.w_d,))(TDL[:, :, :self.N_z])
                    input_control = tf.keras.layers.Reshape((self.N_c * self.w_d,))(TDL[:, :, self.N_z:])

                    output_control = tf.keras.layers.Reshape((self.N_c,))(input[2][:, n, :])

                    input_t = tf.concat([input_feature, input_control, output_control], 1)

                else:
                    TDL = np.concatenate((TDL[:, 1:, :], output_t), axis=1)
                    input_t = tf.keras.layers.Reshape((self.N_z * self.w_d,))(TDL)

                output_t = self.MLP(input_t)
                output[:, n, :] = output_t[:, 0, :]
        else:
            N_steps = int(np.ceil(w_prop / self.w_p))

            for t in range(1, N_steps):

                if self.flag_control:
                    TDL = np.concatenate((TDL[:, self.w_p:, :],
                                          np.concatenate((output_t, input[2][:, (self.w_p * (t - 1)):(self.w_p * t), :]),axis=2)),axis=1)

                    input_feature = tf.keras.layers.Reshape((self.N_z * self.w_d,))(TDL[:, :, :self.N_z])
                    input_control = tf.keras.layers.Reshape((self.N_c * self.w_d,))(TDL[:, :, self.N_z:])

                    if (t == (N_steps - 1)) & ((w_prop / self.w_p) < N_steps):
                        N_last = w_prop - int(np.floor(w_prop / self.w_p)) * self.w_p
                        output_control = tf.keras.layers.Reshape((self.N_c * self.w_p,))(np.concatenate(
                            (input[2][:, (self.w_p * t):, :], input[2][:, -1:, :].repeat(self.w_p - N_last, axis=1)),axis=1))
                    else:
                        output_control = tf.keras.layers.Reshape((self.N_c * self.w_p,))(
                            input[2][:, (self.w_p * t):(self.w_p * (t + 1)), :])

                    input_t = tf.concat([input_feature, input_control, output_control], 1)

                else:

                    TDL = np.concatenate((TDL[:, self.w_p:, :], output_t), axis=1)
                    input_t = tf.keras.layers.Reshape((self.N_z * self.w_d,))(TDL)

                output_t = self.MLP(input_t)

                if (t == (N_steps - 1)) & ((w_prop / self.w_p) < N_steps):
                    N_last = w_prop - int(np.floor(w_prop / self.w_p)) * self.w_p
                    output[:, (self.w_p * t):, :] = output_t[:, :N_last, :]
                else:
                    output[:, (self.w_p * t):(self.w_p * (t + 1)), :] = output_t[:, :self.w_p, :]

        return output

    def call(self, input):

        if self.flag_control:

            TDL = tf.concat([input[0], input[1]], 2)
            input_feature = tf.keras.layers.Reshape((self.N_z * self.w_d,))(input[0])
            input_control = tf.keras.layers.Reshape((self.N_c * self.w_d,))(input[1])

            if not self.flag_recurrent:
                output_control = tf.keras.layers.Reshape((self.N_c * self.w_p,))(input[2][:, :self.w_p, :])
            else:
                output_control = tf.keras.layers.Reshape((self.N_c,))(input[2][:, 0, :])

            input_t = tf.concat([input_feature, input_control, output_control], 1)
        else:

            TDL = tf.identity(input)
            input_t = tf.keras.layers.Reshape((self.N_z * self.w_d,))(TDL)

        output_t = self.MLP(input_t)
        output = tf.identity(output_t)

        if self.flag_recurrent:
            for n in range(1, self.w_p):

                if self.flag_control:

                    TDL = tf.concat([TDL[:, 1:, :], tf.concat([output_t, input[2][:, n - 1:n, :]], 2)], 1)
                    input_feature = tf.keras.layers.Reshape((self.N_z * self.w_d,))(TDL[:, :, :self.N_z])
                    input_control = tf.keras.layers.Reshape((self.N_c * self.w_d,))(TDL[:, :, self.N_z:])

                    output_control = tf.keras.layers.Reshape((self.N_c,))(input[2][:, n, :])

                    input_t = tf.concat([input_feature, input_control, output_control], 1)
                else:
                    TDL = tf.concat([TDL[:, 1:, :], output_t], 1)
                    input_t = tf.keras.layers.Reshape((self.N_z * self.w_d,))(TDL)

                output_t = self.MLP(input_t)
                output = tf.concat([output, output_t], 1)

        return output