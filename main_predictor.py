import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping

from utils.data.transform_data import raw2dyn
from utils.dynamics.predictors_lm import NARX, LSTM, MeanSquaredError
from utils.dynamics.systems import get_lorenz_time

# PARAMETERS AND FLAGS
params = {}
params['AE'] = {}
params['dyn'] = {}
params['flow'] = {}
params['LSTM'] = {}
params['NARX'] = {}

flags = {}
flags['dyn'] = {}

params['AE']['nr'] = 3
params['AE']['n_epochs'] = 500
params['AE']['batch_size'] = 5000
params['AE']['lr'] = 1e-3

params['dyn']['np'] = 10
params['dyn']['d'] = 10
params['dyn']['act'] = 'tanh'
params['dyn']['nt_pred'] = 500

params['dyn']['dt'] = 0.01
params['dyn']['dt_lat'] = 0.01
params['dyn']['irreg'] = 0

params['dyn']['o'] = params['dyn']['np'] + params['dyn']['d']

params['dyn']['kreg'] = 0

params['LSTM']['lstm_units'] = [16]
params['LSTM']['dense_units'] = []
params['LSTM']['dropout'] = []

params['NARX']['units'] = [10]
params['NARX']['dropout'] = [0]

flags['dyn']['control'] = 1
flags['dyn']['type'] = 'LSTM'
flags['dyn']['multi_train'] = 0

params['flow']['nc'] = 1 if flags['dyn']['control'] else 0

# GENERATE TIME CHAOTIC SEQUENCES
sigma, rho, beta = 10, 28, 8/3
t_span = np.linspace(0, 10, int(10 / (params['dyn']['dt']) +1))
nt = len(t_span)
u = (np.sin(t_span)).reshape((nt, 1))
n_sets = 10

for i in range(n_sets):
    Y0 = np.random.rand(params['AE']['nr']) * 30 - 15

    if not flags['dyn']['control']:
        Y = get_lorenz_time(t_span, Y0, sigma, beta, rho)

        zx_train, zy_train, zx_val, zy_val = raw2dyn(t_span, Y, params, flags['dyn']['control'])

        if i == 0:
            Zx_train, Zy_train, Zx_val, Zy_val = zx_train, zy_train, zx_val, zy_val
        else:
            Zx_train, Zy_train, Zx_val, Zy_val = np.concatenate((Zx_train, zx_train), axis=0), np.concatenate((Zy_train, zy_train), axis=0),\
                                                 np.concatenate((Zx_val, zx_val), axis=0), np.concatenate((Zy_val, zy_val), axis=0)
    else:
        Y = get_lorenz_time(t_span, Y0, sigma, beta, rho, b_span=u, flag_control=1)

        zx_train, zy_train, zx_val, zy_val, ux_train, uy_train, ux_val, uy_val = raw2dyn(t_span, Y, params, flags['dyn']['control'], u = u)

        if i == 0:
            Zx_train, Zy_train, Zx_val, Zy_val = zx_train, zy_train, zx_val, zy_val
            Ux_train, Uy_train, Ux_val, Uy_val = ux_train, uy_train, ux_val, uy_val
        else:
            Zx_train, Zy_train, Zx_val, Zy_val = np.concatenate((Zx_train, zx_train), axis=0), np.concatenate(
                (Zy_train, zy_train), axis=0), \
                np.concatenate((Zx_val, zx_val), axis=0), np.concatenate((Zy_val, zy_val), axis=0)
            Ux_train, Uy_train, Ux_val, Uy_val = np.concatenate((Ux_train, ux_train), axis=0), np.concatenate(
                (Uy_train, uy_train), axis=0), \
                np.concatenate((Ux_val, ux_val), axis=0), np.concatenate((Uy_val, uy_val), axis=0)

Ynorm = np.max(np.abs(Zx_train), axis=(0,1)).reshape(1,1,-1)
Zx_train, Zy_train, Zx_val, Zy_val = Zx_train / Ynorm, Zy_train / Ynorm, Zx_val / Ynorm, Zy_val / Ynorm

# TRAIN DYNAMIC MODEL
ES = EarlyStopping(monitor="val_loss", patience=50)

if flags['dyn']['type'] == 'NARX':
    DYN = NARX(params, flags)
else:
    DYN = LSTM(params, flags)

DYN.compile(tf.keras.optimizers.SGD(learning_rate=1.0),
    loss=MeanSquaredError(), run_eagerly=True)

if not flags['dyn']['control']:
    DYN.fit(Zx_train, Zy_train,
             epochs=params['AE']['n_epochs'],
             shuffle=True,
             validation_data=(Zx_val, Zy_val),
             batch_size=params['AE']['batch_size'],
             verbose=1,
             callbacks=[ES])
else:
    DYN.fit([Zx_train, Ux_train, Uy_train], Zy_train,
            epochs=params['AE']['n_epochs'],
            shuffle=True,
            validation_data=([Zx_val, Ux_val, Uy_val], Zy_val),
            batch_size=params['AE']['batch_size'],
            verbose=1,
            callbacks=[ES])

# GENERATE TESTING SET & PREDICT
Y0 = np.random.rand(params['AE']['nr']) * 30 - 15

if not flags['dyn']['control']:
    Y = get_lorenz_time(t_span, Y0, sigma, beta, rho)

    Zx_test, Zy_test, T = raw2dyn(t_span, Y, params, flags['dyn']['control'], flag_train=0)

    Zx_test, Zy_test = Zx_test / Ynorm, Zy_test / Ynorm
    Zy_test_dyn = DYN.predict(Zx_test,params['dyn']['nt_pred'])
else:
    Y = get_lorenz_time(t_span, Y0, sigma, beta, rho, b_span=u, flag_control=1)

    Zx_test, Zy_test, Ux_test, Uy_test, T = raw2dyn(t_span, Y, params, flags['dyn']['control'], flag_train=0, u=u)
    Zx_test, Zy_test = Zx_test / Ynorm, Zy_test / Ynorm
    Zy_test_dyn = DYN.predict([Zx_test, Ux_test, Uy_test], params['dyn']['nt_pred'])

# PLOT
Y = Y / Ynorm[0,0,:]

nrows = 1
ncols = params['AE']['nr']
fig, ax = plt.subplots(nrows,ncols, subplot_kw=dict(box_aspect=1))

w = 0
i,j = 0,0
for c, ax in enumerate(fig.axes):

    if j == ncols:
        i += 1
        j = 0

    ax.plot([T['TDL_lat'][w][0], T['pred'][w][-1]], [0, 0], '-', color='gray', linewidth=0.8,
            label='_nolegend_')
    ax.plot(T['TDL_lat'][w], Zx_test[w, :, c], 'y-', linewidth=1.3, label='TDL')
    ax.plot(T['pred'][w], Zy_test[w, :, c], 'k-', linewidth=1.3, label='ground truth')
    if params['dyn']['np'] > 5:
        ax.plot(T['pred'][w][0:params['dyn']['np']], Zy_test_dyn[w,0:params['dyn']['np'], c], 'b-.', linewidth=1.3,
                label='pred window')
    else:
        ax.plot(T['pred'][w][0:params['dyn']['np']], Zy_test_dyn[w,0:params['dyn']['np'], c], 'b.-', linewidth=1.3,
                label='pred window')
    ax.plot(T['pred'][w][params['dyn']['np']:], Zy_test_dyn[w,params['dyn']['np']:, c], 'r--', linewidth=1.3,
            label='pred propagation')

    ax.text(T['TDL_lat'][w][0],-0.95, '$z_{' + str(c) + '}$', color='r', fontsize=12)

    ax.set_ylim([-1, 1])
    ax.set_xlim([T['TDL_lat'][w][0], T['pred'][w][-1]])

    if i == (nrows -1):
        ax.set_xlabel('$t$ [s]')
        ax.set_xticks([T['TDL_lat'][w][0],T['pred'][w][-1]])
    else:
        ax.set_xticks([])

    if j == 0:
        ax.set_ylabel('$z$')
        ax.set_yticks([-1,0,1])
    else:
        ax.set_yticks([])

    if c == 0:
        ax.legend()

    j += 1

plt.show()

a=0

