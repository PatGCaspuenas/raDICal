# PACKAGES
import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import numpy as np
import re
import random


# PARAMETERS

path_out = r'F:\AEs_wControl\misc\1st_dyn'
path_list = r'F:\AEs_wControl\OUTPUT\dyn_1st_control.csv'

files = os.listdir(path_out)
files = [f for f in files if f.endswith('.mat')]

df = pd.read_csv(path_list)

df['l_train_f'] = 0
df['l_val_f'] = 0
df['RMSE_dyn'] = 0
df['NT'] = 0
df['R2C'] = 0
df['Dtime'] = 0

df['RMSE_AE_np'] = 0
df['CEA_np'] = 0
df['Sc_np'] = 0
df['RMSE_AE_nt'] = 0
df['CEA_nt'] = 0
df['Sc_nt'] = 0

for f in files:
    # Get hyperparameters of file
    vars = re.split('_|c|lr|nepoch|batch|beta|nr|nt|fc|lstmu|lstmdu|narxdu|np|d|history.mat', f)
    vars = list(filter(None, vars))
    DYN, AE, lr, n_epochs, batch_size, fc, d, n_p, lstmu, lstmdu, narxdu = vars
    lr, n_epochs, batch_size, d, fc, n_p = float(lr), int(float(n_epochs)), int(float(batch_size)), int(float(
        d)), int(float(fc)), int(float(n_p))

    # Match with csv list
    ilist = df.loc[(df['lr_d'] == lr) & (df['n_epochs_d'] == n_epochs) & (df['batch_size_d'] == batch_size)
                   & (df['flag_control_dyn'] == fc) & (df['d'] == d) & (df['np'] == n_p) &
                   (df['lstm_units'] == lstmu) & (df['dense_units'] == lstmdu) & (df['units'] == narxdu) &
                   (df['AE'] == AE) & (df['dyn'] == DYN)].index[0]

    M = sio.loadmat(os.path.join(path_out, f))

    df['l_train_f'][ilist] = M['loss'][0, -1]
    df['l_val_f'][ilist] = M['val_loss'][0, -1]
    df['RMSE_dyn'][ilist] = np.mean(M['RMSE_dyn'], axis=1)
    df['NT'][ilist] = M['NT'][0][0]
    df['R2C'][ilist] = np.mean(M['R2C'], axis=1)
    df['Dtime'][ilist] = M['Dtime'][0][0] / 60

    # df['CEA_np'][ilist] = M['CEA_np'][0][0]
    # df['Sc_np'][ilist] = M['Sc_np'][0][0]
    # df['RMSE_AE_np'][ilist] = M['RMSE_AE_np'][0][0]
    # df['CEA_nt'][ilist] = M['CEA_nt'][0][0]
    # df['Sc_nt'][ilist] = M['Sc_nt'][0][0]
    # df['RMSE_AE_nt'][ilist] = M['RMSE_AE_nt'][0][0]

a=0

