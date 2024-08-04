import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

path_cl = r'F:\Re150\ClValues'
path_cd = r'F:\Re150\CdValues'
path_input = r'F:\Re150\InputValues'
inputs = 3

for i in range(1, inputs + 1):
    df = pd.read_csv(path_cl + str(i) + '.txt', header=None, sep='\t')
    if i == 1:
        cltrain = df.to_numpy()[:, 1:]
        t_train = df.to_numpy()[:, 0] - 0.1
    else:
        cltrain = np.concatenate((cltrain, df.to_numpy()[:, 1:]), axis=0)
        t_train = np.concatenate((t_train, df.to_numpy()[:, 0] - 0.1), axis=0)
cltrain = np.concatenate((cltrain, (cltrain[:, 0] + cltrain[:, 1] + cltrain[:, 2]).reshape(-1, 1)), axis=1)

df = pd.read_csv(path_cl + str(4) + '.txt', header=None, sep='\t')
clval = df.to_numpy()[:, 1:]
clval = np.concatenate((clval, (clval[:, 0] + clval[:, 1] + clval[:, 2]).reshape(-1, 1)), axis=1)
t_val = df.to_numpy()[:, 0] - 0.1

for i in range(1, inputs + 1):
    df = pd.read_csv(path_cd + str(i) + '.txt', header=None, sep='\t')
    if i == 1:
        cdtrain = df.to_numpy()[:, 1:]
    else:
        cdtrain = np.concatenate((cdtrain, df.to_numpy()[:, 1:]), axis=0)
cdtrain = np.concatenate((cdtrain, (cdtrain[:, 0] + cdtrain[:, 1] + cdtrain[:, 2]).reshape(-1, 1)), axis=1)

df = pd.read_csv(path_cd + str(4) + '.txt', header=None, sep='\t')
cdval = df.to_numpy()[:, 1:]
cdval = np.concatenate((cdval, (cdval[:, 0] + cdval[:, 1] + cdval[:, 2]).reshape(-1, 1)), axis=1)

for i in range(1, inputs + 1):
    df = pd.read_csv(path_input + str(i) + '.txt', header=None, sep=' ')
    if i == 1:
        omega_train = df.to_numpy()
    else:
        omega_train = np.concatenate((omega_train, df.to_numpy()), axis=0)

df = pd.read_csv(path_input + str(4) + '.txt', header=None, sep=' ')
omega_val = df.to_numpy()

t_train = np.concatenate((np.array([0, 0.1]), t_train), axis=0)
t_val = np.concatenate((np.array([0]), t_val), axis=0)

omega_val = np.concatenate((np.zeros((1, 3)), omega_val), axis=0)
omega_train = np.concatenate((np.zeros((2, 3)), omega_train), axis=0)

cdval = np.concatenate((np.zeros((1, 4)), cdval), axis=0)
cdtrain = np.concatenate((np.zeros((2, 4)), cdtrain), axis=0)

clval = np.concatenate((np.zeros((1, 4)), clval), axis=0)
cltrain = np.concatenate((np.zeros((2, 4)), cltrain), axis=0)

flag_type = 'val'
ii = np.arange(0, np.shape(clval)[0])
if flag_type == 'train':
    forces = {}
    forces['t'] = t_train[ii].reshape(-1, 1)
    forces['vF'] = omega_train[ii, 0].reshape(-1, 1)
    forces['vT'] = omega_train[ii, 1].reshape(-1, 1)
    forces['vB'] = omega_train[ii, 2].reshape(-1, 1)

    forces['Cl_F'] = cltrain[ii, 0].reshape(-1, 1)
    forces['Cl_T'] = cltrain[ii, 1].reshape(-1, 1)
    forces['Cl_B'] = cltrain[ii, 2].reshape(-1, 1)
    forces['Cl_total'] = cltrain[ii, 3].reshape(-1, 1)

    forces['Cd_F'] = cdtrain[ii, 0].reshape(-1, 1)
    forces['Cd_T'] = cdtrain[ii, 1].reshape(-1, 1)
    forces['Cd_B'] = cdtrain[ii, 2].reshape(-1, 1)
    forces['Cd_total'] = cdtrain[ii, 3].reshape(-1, 1)
else:
    forces = {}
    forces['t'] = t_val[ii].reshape(-1, 1)
    forces['vF'] = omega_val[ii, 0].reshape(-1, 1)
    forces['vT'] = omega_val[ii, 1].reshape(-1, 1)
    forces['vB'] = omega_val[ii, 2].reshape(-1, 1)

    forces['Cl_F'] = clval[ii, 0].reshape(-1, 1)
    forces['Cl_T'] = clval[ii, 1].reshape(-1, 1)
    forces['Cl_B'] = clval[ii, 2].reshape(-1, 1)
    forces['Cl_total'] = clval[ii, 3].reshape(-1, 1)

    forces['Cd_F'] = cdval[ii, 0].reshape(-1, 1)
    forces['Cd_T'] = cdval[ii, 1].reshape(-1, 1)
    forces['Cd_B'] = cdval[ii, 2].reshape(-1, 1)
    forces['Cd_total'] = cdval[ii, 3].reshape(-1, 1)

path_forces = r'F:\AEs_wControl\DATA\forces\FPcf_00k_03k.h5'
with h5py.File(path_forces, 'w') as h5file:
    for key, item in forces.items():
        h5file.create_dataset(key, data=item)

a=0