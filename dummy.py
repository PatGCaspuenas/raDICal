# PACKAGES
import os
import h5py

import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import numpy as np
import re
import random

# PARAMETERS

path_out = r'F:\AEs_wControl\misc\2nd_dyn'
path_list = r'F:\AEs_wControl\OUTPUT\dyn_2nd_control.csv'

files = os.listdir(path_out)
files = [f for f in files if f.endswith('.mat')]

df = pd.read_csv(path_list)

df['l_train_f'] = 0
df['l_val_f'] = 0
df['RMSE_dyn'] = 0
df['RMSE_dyn_ntpred'] = 0

df['R2C'] = 0
df['R2C_ntpred'] = 0

df['R2C_5'] = 0
df['R2C_20'] = 0
df['R2C_50'] = 0
df['R2C_100'] = 0
df['R2C_200'] = 0

df['Dtime'] = 0
df['Dt_pred'] = 0
df['Dt_train'] = 0
df['NT'] = 0

OUT = list(tuple([{}]*len(files)))

for f in files:
    # Read out
    out = {}
    with h5py.File(path_out + '\\' + f[:-11] + 'out.h5', 'r') as f1:
        for i in f1.keys():
            out[i] = f1[i][()]
    OUT.append(out)

    # Get hyperparameters of file
    vars = re.split('c|lr|nepoch|batch|beta|nr|nt|fc|lstmu|lstmdu|narxdu|np|d_|_|opt|history.mat', f)
    vars = list(filter(None, vars))
    DYN, AE, lr, n_epochs, batch_size, fc, d, n_p, lstmu, lstmdu, narxdu, opt = vars
    lr, n_epochs, batch_size, d, fc, n_p = float(lr), int(float(n_epochs)), int(float(batch_size)), int(float(
        d)), int(float(fc)), int(float(n_p))

    # Match with csv list
    ilist = df.loc[(df['lr_d'] == lr) & (df['n_epochs_d'] == n_epochs) & (df['batch_size_d'] == batch_size)
                   & (df['flag_control_dyn'] == fc) & (df['d'] == d) & (df['np'] == n_p) &
                   (df['lstm_units'] == lstmu) & (df['dense_units'] == lstmdu) & (df['units'] == narxdu) &
                   (df['AE'] == AE) & (df['dyn'] == DYN) & (df['opt'] == opt)].index[0]

    M = sio.loadmat(os.path.join(path_out, f))

    df['l_train_f'][ilist] = M['loss'][0, -1]
    df['l_val_f'][ilist] = M['val_loss'][0, -1]
    df['RMSE_dyn'][ilist] = np.mean(M['RMSE_dyn'], axis=1)
    df['RMSE_dyn_ntpred'][ilist] = np.mean(M['RMSE_dyn_ntpred'], axis=1)

    df['R2C'][ilist] = np.mean(M['R2C'], axis=1)
    df['R2C_ntpred'][ilist] = np.mean(M['R2C_ntpred'], axis=1)

    df['R2C_5'][ilist] = np.mean(M['R2C_5'], axis=1)
    df['R2C_20'][ilist] = np.mean(M['R2C_20'], axis=1)
    df['R2C_50'][ilist] = np.mean(M['R2C_50'], axis=1)
    df['R2C_100'][ilist] = np.mean(M['R2C_100'], axis=1)
    df['R2C_200'][ilist] = np.mean(M['R2C_200'], axis=1)

    df['NT'][ilist] = M['NT'][0][0]
    df['Dtime'][ilist] = M['Dtime'][0][0] / 60
    df['Dt_pred'][ilist] = M['Dt_pred'][0][0] / 60
    df['Dt_train'][ilist] = M['Dt_train'][0][0] / 60

    # df['CEA_np'][ilist] = M['CEA_np'][0][0]
    # df['Sc_np'][ilist] = M['Sc_np'][0][0]
    # df['RMSE_AE_np'][ilist] = M['RMSE_AE_np'][0][0]
    # df['CEA_nt'][ilist] = M['CEA_nt'][0][0]
    # df['Sc_nt'][ilist] = M['Sc_nt'][0][0]
    # df['RMSE_AE_nt'][ilist] = M['RMSE_AE_nt'][0][0]

    OUT[ilist] = out

a=0

iout = 3

nrows = 5
ncols = 4
fig, ax = plt.subplots(nrows,ncols, subplot_kw=dict(box_aspect=1))

w = 0
i,j = 0,0
for c, ax in enumerate(fig.axes):

    if j == ncols:
        i += 1
        j = 0

    ax.plot([OUT[iout]['t_test'][w,0,0], OUT[iout]['t_test'][w,-1,0]], [0, 0], '-', color='gray', linewidth=0.8,
            label='_nolegend_')
    ax.plot(OUT[iout]['t_test'][w,:,0], OUT[iout]['Zy_test'][w, :, c], 'k-', linewidth=1.3, label='ground truth')
    ax.plot(OUT[iout]['t_test'][w,:,0], OUT[iout]['Zy_dyn_test'][w, :, c], 'r--', linewidth=1.3,
            label='pred ')

    ax.text(OUT[iout]['t_test'][w,0,0],-0.95, '$z_{' + str(c) + '}$', color='r', fontsize=12)

    ax.set_ylim([-1, 1])
    ax.set_xlim([OUT[iout]['t_test'][w,0,0], OUT[iout]['t_test'][w,-1,0]])

    if i == (nrows -1):
        ax.set_xlabel('$t$ [s]')
        ax.set_xticks([OUT[iout]['t_test'][w,0,0], OUT[iout]['t_test'][w,-1,0]])
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