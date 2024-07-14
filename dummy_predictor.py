import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.dynamics.systems import get_lorenz_time
from utils.others import levenberg_marquardt as lm

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
params['AE']['batch_size'] = 32
params['AE']['lr'] = 1e-3

params['dyn']['np'] = 1
params['dyn']['d'] = 1
params['dyn']['act'] = 'tanh'
params['dyn']['nt_pred'] = 5000

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

flags['dyn']['control'] = 0
flags['dyn']['type'] = 'NARX'

params['flow']['nc'] = 1 if flags['dyn']['control'] else 0

# GENERATE TIME CHAOTIC SEQUENCES
sigma, rho, beta = 10, 28, 8/3
t_span = np.linspace(0, 110, int(110 / (params['dyn']['dt']) +1))
nt = len(t_span)
u = (np.sin(t_span)).reshape((nt, 1))

Y0 = np.array([-8,8,27]) #np.random.rand(params['AE']['nr']) * 30 - 15

Y = get_lorenz_time(t_span, Y0, sigma, beta, rho)

zx_train, zy_train, zx_val, zy_val = Y[0:10000,:], Y[1:10001,:], Y[10000:-1,:], Y[10001:,:]

Zx_train, Zy_train, Zx_val, Zy_val = zx_train.reshape(-1,1,params['AE']['nr']), zy_train.reshape(-1,1,params['AE']['nr']), zx_val.reshape(-1,1,params['AE']['nr']), zy_val.reshape(-1,1,params['AE']['nr'])

nt_train = np.shape(Zx_train)[0]

Ynorm = np.max(np.abs(Zx_train), axis=(0,1)).reshape(1,1,-1)
Zx_train, Zy_train, Zx_val, Zy_val = Zx_train / Ynorm, Zy_train / Ynorm, Zx_val / Ynorm, Zy_val / Ynorm

#ES = EarlyStopping(monitor="val_loss", patience=100)

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Dense(10, activation=params['dyn']['act'], input_shape=([params['dyn']['d'] * params['AE']['nr']])),
                                    tf.keras.layers.Dense(params['AE']['nr'], activation='linear'),
        ])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mse')

model_wrapper = lm.ModelWrapper(tf.keras.models.clone_model(model))

model_wrapper.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1),
    loss=lm.MeanSquaredError())

model.fit(Zx_train[:,0,:],
                  Zy_train[:,0,:], epochs=1000, batch_size=nt_train)

model_wrapper.fit(Zx_train[:,0,:],
                  Zy_train[:,0,:], epochs=1000, batch_size=nt_train)

# GENERATE TESTING SET & PREDICT

nt = np.shape(Zx_val)[0]
tspan_val = np.linspace(0,nt * 0.01 - 0.01,nt)
Zy_val_dyn = np.zeros((nt,1,3))
Zy_val_dyn_wrap = np.zeros((nt,1,3))
for t in range(nt):
    if t == 0:
        Zy_val_dyn[t,0,:] = model(Zx_val[t:t+1,0,:])
        Zy_val_dyn_wrap[t, 0, :] = model_wrapper(Zx_val[t:t+1, 0, :])
    else:
        Zy_val_dyn[t,0,:] = model(Zy_val_dyn[t-1:t,0,:])
        Zy_val_dyn_wrap[t, 0, :] = model_wrapper(Zy_val_dyn_wrap[t-1:t, 0, :])

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


    ax.plot(tspan_val, Zy_val[:, 0, c], 'k-', linewidth=1.3, label='ground truth')
    ax.plot(tspan_val, Zy_val_dyn[:, 0, c], 'r--', linewidth=1.3, label='Adam')
    ax.plot(tspan_val, Zy_val_dyn_wrap[:, 0, c], 'b--', linewidth=1.3, label='LM')


    #ax.set_ylim([-1, 1])

    if i == (nrows -1):
        ax.set_xlabel('$t$ [s]')
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