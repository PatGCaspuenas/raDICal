# PACKAGES
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping
from timeit import default_timer as timer

# LOCAL FUNCTIONS
from utils.data.transform_data import raw2dyn
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

def train_dyn(params, flags, Z, t, logging, u=0):

    # PARAMETERS
    nt, nr = np.shape(Z)
    Znorm = np.max(np.abs(Z), axis=0).reshape(1, 1, -1)

    lr = params['dyn']['lr']
    n_epochs = params['dyn']['n_epochs']
    batch_size = params['dyn']['batch_size']

    logger = MyLogger(logging, n_epochs)
    ES = EarlyStopping(monitor="val_loss", patience=100)

    # FLAGS
    flag_control = flags['dyn']['control']
    flag_type = flags['dyn']['type']
    flag_opt = flags['dyn']['opt']
    flag_loss = flags['dyn']['loss']

    # GENERATE WINDOW PREDICTIONS
    if not flag_control:
        Zx_train, Zy_train, Zx_val, Zy_val = raw2dyn(t, Z, params, flag_control)
    else:
        Zx_train, Zy_train, Zx_val, Zy_val, Ux_train, Uy_train, Ux_val, Uy_val = raw2dyn(t, Z, params, flag_control, u=u)

    # NORMALIZE STATE VARIABLES
    #Zx_train, Zy_train, Zx_val, Zy_val = Zx_train / Znorm, Zy_train / Znorm, Zx_val / Znorm, Zy_val / Znorm

    # TRAIN DYNAMIC MODEL
    nw_train, nw_val = np.shape(Zx_train)[0], np.shape(Zx_val)[0]
    logging.info(f'{nw_train} training windows, {nw_val} validation windows')

    if flag_opt == 'Adam':
        from utils.dynamics.predictors import NARX, LSTM
    else:
        from utils.dynamics.predictors_lm import NARX, LSTM, MeanSquaredError

    if flag_type == 'NARX':
        DYN = NARX(params, flags)
    else:
        DYN = LSTM(params, flags)

    if flag_opt == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        if flag_loss == 'mse':
            DYN.compile(optimizer=opt, loss='mse', metrics = ['mae'])
        else:
            DYN.compile(optimizer=opt, loss=tf.keras.losses.Huber(), metrics = ['mae'])
    else:
        DYN.compile(tf.keras.optimizers.SGD(learning_rate=1.0),
                    loss=MeanSquaredError(), run_eagerly=True)

    if not flags['dyn']['control']:
        DYN.fit(Zx_train, Zy_train,
                epochs=n_epochs,
                shuffle=True,
                validation_data=(Zx_val, Zy_val),
                batch_size=batch_size,
                verbose=2,
                callbacks=[logger, ES])

    else:
        DYN.fit([Zx_train, Ux_train, Uy_train], Zy_train,
                epochs=n_epochs,
                shuffle=True,
                validation_data=([Zx_val, Ux_val, Uy_val], Zy_val),
                batch_size=batch_size,
                verbose=2,
                callbacks=[logger, ES])

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

    return DYN, Znorm





