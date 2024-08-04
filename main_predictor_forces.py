import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn import preprocessing

from utils.data.transform_data import raw2dyn
#from utils.dynamics.predictors_lm import NARX, LSTM, MeanSquaredError
from utils.dynamics.predictors import NARX, LSTM
import h5py
import random

def R2_loss(input,output):
    nr = tf.shape(input)[2]
    R2 = output[:, 0,0] * 0
    for i in range(nr):
        R2nom = tf.keras.backend.mean(tf.keras.backend.square(tf.keras.layers.Multiply()([output[:, :, i], input[:, :, i]])), axis=1)
        R2den = tf.keras.layers.Multiply()([tf.keras.backend.mean(tf.keras.backend.square(output[:, :, i]), axis=1), tf.keras.backend.mean(tf.keras.backend.square(input[:, :, i]), axis=1)])
        R2den = R2den + 1e-4
        R2 = R2 + tf.math.divide(R2nom,R2den)

    R2_loss = float(nr) - tf.keras.backend.mean(R2)

    return R2_loss
# PARAMETERS AND FLAGS
params = {}
params['AE'] = {}
params['dyn'] = {}
params['flow'] = {}
params['LSTM'] = {}
params['NARX'] = {}

flags = {}
flags['dyn'] = {}

params['AE']['nr'] = 6
params['AE']['n_epochs'] = 1000
params['AE']['batch_size'] = 10000
params['AE']['lr'] = 1

params['dyn']['np'] = 20
params['dyn']['d'] = 20
params['dyn']['act'] = 'tanh'
params['dyn']['nt_pred'] = 1000

params['dyn']['dt'] = 0.1
params['dyn']['dt_lat'] = 0.1
params['dyn']['irreg'] = 0

params['dyn']['o'] = 1

params['dyn']['kreg'] = 0

params['LSTM']['lstm_units'] = [64]
params['LSTM']['dense_units'] = [128]
params['LSTM']['dropout'] = [0.3]

params['NARX']['units'] = [128]
params['NARX']['dropout'] = [0.3]

flags['dyn']['control'] = 1
flags['dyn']['type'] = 'LSTM'
flags['dyn']['multi_train'] = 0

params['flow']['nc'] = 3 if flags['dyn']['control'] else 0

# GENERATE TIME CHAOTIC SEQUENCES
path_forces = r'F:\AEs_wControl\DATA\forces\FPcf_00k_70k.h5'
forces = {}
with h5py.File(path_forces, 'r') as f:
    for i in f.keys():
        forces[i] = f[i][()]
path_forces_test = r'F:\AEs_wControl\DATA\forces\FPcf_00k_03k.h5'
forces_test = {}
with h5py.File(path_forces_test, 'r') as f:
    for i in f.keys():
        forces_test[i] = f[i][()]


Z = np.concatenate((forces['Cl_F'],forces['Cl_T'],forces['Cl_B']),axis=1)
Z = np.concatenate((forces['Cl_total'],forces['Cd_total']),axis=1)
Z = np.concatenate((forces['Cl_F'],forces['Cl_T'],forces['Cl_B'],forces['Cd_F'],forces['Cd_T'],forces['Cd_B']),axis=1)


t = forces['t']
U = np.concatenate((forces['vF'],forces['vT'],forces['vB']),axis=1)
Z, U, t = Z[2:,:],U[2:,:],t[2:,:]

Z_test = np.concatenate((forces_test['Cl_F'],forces_test['Cl_T'],forces_test['Cl_B']),axis=1)
Z_test = np.concatenate((forces_test['Cl_total'],forces_test['Cd_total']),axis=1)
Z_test = np.concatenate((forces_test['Cl_F'],forces_test['Cl_T'],forces_test['Cl_B'],forces_test['Cd_F'],forces_test['Cd_T'],forces_test['Cd_B']),axis=1)



t_test = forces_test['t']
U_test = np.concatenate((forces_test['vF'],forces_test['vT'],forces_test['vB']),axis=1)
Z_test, U_test, t_test = Z_test[2:,:],U_test[2:,:],t_test[2:,:]
# NORMALIZE
scaler_Z = preprocessing.RobustScaler()
scaler_U = preprocessing.RobustScaler()
Z_unscaled = Z
Z_test_unscaled = Z_test
U_unscaled = U
U_test_unscaled = U_test

Z = scaler_Z.fit_transform(Z)
Z_test = scaler_Z.transform(Z_test)
U = scaler_U.fit_transform(U)
U_test = scaler_U.transform(U_test)


if not flags['dyn']['control']:
    Zx_train, Zy_train, Zx_val, Zy_val = raw2dyn(t, Z, params, flags['dyn']['control'])

else:
    Zx_train, Zy_train, Zx_val, Zy_val, Ux_train, Uy_train, Ux_val, Uy_val = raw2dyn(t, Z, params, flags['dyn']['control'], u = U)

#Ynorm = np.max(np.abs(Zx_train), axis=(0,1)).reshape(1,1,-1)
#Zx_train, Zy_train, Zx_val, Zy_val = Zx_train / Ynorm, Zy_train / Ynorm, Zx_val / Ynorm, Zy_val / Ynorm

#flags['dyn']['control'] = 0
#params['flow']['nc'] = 1 if flags['dyn']['control'] else 0
# TRAIN DYNAMIC MODEL
ES = EarlyStopping(monitor="val_loss", patience=50)

if flags['dyn']['type'] == 'NARX':
    DYN = NARX(params, flags)
else:
    DYN = LSTM(params, flags)

#DYN.compile(tf.keras.optimizers.SGD(learning_rate=1.0),loss=MeanSquaredError(), run_eagerly=True)
DYN.compile(optimizer='Adam',loss=tf.keras.losses.Huber(),metrics=['mae'])

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

if not flags['dyn']['control']:

    Zx_test, Zy_test, T = raw2dyn(t_test, Z_test, params, flags['dyn']['control'], flag_train=0)

    #Zx_test, Zy_test = Zx_test / Ynorm, Zy_test / Ynorm
    Zy_test_dyn = DYN.predict(Zx_test,params['dyn']['nt_pred'])
else:

    Zx_test, Zy_test, Ux_test, Uy_test, T = raw2dyn(t_test, Z_test, params, flags['dyn']['control'], flag_train=0, u=U_test)
    #Zx_test, Zy_test = Zx_test / Ynorm, Zy_test / Ynorm
    Zy_test_dyn = DYN.predict([Zx_test, Ux_test, Uy_test], params['dyn']['nt_pred'])

# PLOT


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
        ax.set_xticks([T['TDL_lat'][w][0,0],T['pred'][w][-1,0]])
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
