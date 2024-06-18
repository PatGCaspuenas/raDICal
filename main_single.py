# GPU REQUIREMENTS
import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
available_GPUs = len(physical_devices)
print('Using TensorFlow version: ', tf.__version__, ', GPU: ', available_GPUs)
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# PACKAGES
import warnings
import scipy.io as sio
import matplotlib.pyplot as plt
import random
import h5py
import logging
import time

# LOCAL FILES
from utils.data.config import log_initial_config
from utils.data.read_data import read_FP, read_SC
from utils.data.transform_data import get_control_vector

from utils.AEs.modes import get_modes_AE, get_correlation_matrix
from utils.AEs.energy import energy_AE, energy_POD
from utils.AEs.train import train_AE
from utils.AEs.outputs import get_AE_z, get_AE_reconstruction
from utils.modelling.errors import get_RMSE, get_CEA, get_cos_similarity, get_latent_correlation_matrix

from utils.plt.plt_snps import *
from utils.modelling.differentiation import get_2Dvorticity

# FLAGS
flags = {}
flags['save'] = {}

flags['AE'] = 'C-CNN-AE'  # AE type (C-CNN-AE, MD-CNN-AE, CNN-HAE, CNN-VAE)
flags['struct'] = 'complex'  # AE structure type (simple, medium, complex)
flags['flow'] = 'FP'  # Flow type (SC, FP)
flags['control'] = 1  # With (1) or without (0) control
flags['POD'] = 0  # Gets POD basis (1) or not (0)
flags['filter'] = 0
flags['lr_static'] = 1  # Fixed learning rate (1) or varying with nepochs (0)

flags['get_modal'] = 0  # Retrieve modal analysis
flags['get_reconstruction'] = 1  # Retrieve AE reconstruction and error quantification
flags['get_latent'] = 0
flags['save']['model'] = 0  # Save trained model (1) or not (0)
flags['save']['out'] = 0  # Save outputs of AE (1) or not (0)
flags['save']['history'] = 1  # Save model history loss (1) or not (0)
flags['error_type'] = 'W'  # Type of RMSE

# PARAMETERS
params = {}
params['AE'] = {}
params['flow'] = {}
params['POD'] = {}

# Learning rate in Adam optimizer
if flags['lr_static']:
    params['AE']['lr'] = 1e-5
else:
    step = tf.Variable(0, trainable=False)
    boundaries = [500, 100, 100]
    values = [1e-3, 1e-4, 1e-5, 1e-6]
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    params['AE']['lr'] = lr(step)
    del step, boundaries, values

params['AE']['logger'] = 'HOLI2C-CNN-AE' + '_lr_' + '1e-5' + \
                         '_nepoch_' + '500' + \
                         '_batch_' + '64' + \
                         '_beta_' + '0' + \
                         '_nr_' + '15' + \
                         '_nt_' + '10000'  # Logger name

params['AE']['n_epochs'] = 500  # Number of epochs
params['AE']['batch_size'] = 64  # Batch size when training AE
params['AE']['beta'] = 5e-3  # CNN-VAE reg parameter
params['AE']['ksize'] = (3, 3)  # Kernel size of convolutional layers
params['AE']['psize'] = (2, 2)  # Kernel size of pooling layers
params['AE']['ptypepool'] = 'valid'  # Type of pooling padding (same or valid)
params['AE']['nstrides'] = 2  # Number of strides in pooling
params['AE']['act'] = 'tanh'  # Activation function
params['AE']['reg_k'] = 0  # Regularization kernel
params['AE']['reg_b'] = 0  # Regularization bias
params['AE']['nr'] = 15  # AE latent space dimensions

params['flow']['k'] = 2  # Number of dimensions
params['flow']['nt'] = 10000

params['POD']['nt'] = 500  # Number of snapshots to create POD basis
params['POD']['nr'] = params['AE']['nr']  # Number of modes in truncated basis

# PATHS
cwd = os.getcwd()
paths = {}

paths['history'] = cwd + r'/OUTPUT/' + params['AE']['logger'] + '_history.mat'
paths['output'] = cwd + r'/OUTPUT/' + params['AE']['logger'] + '_out.h5'
paths['model'] = cwd + r'/MODELS/' + params['AE']['logger'] + '.weights.h5'
paths['logger'] = cwd + r'/OUTPUT/' + params['AE']['logger'] + '_logger.log'

if flags['flow'] == 'FP':
    paths['grid'] = cwd + r'/DATA/FP_grid.h5'

    if flags['control']:
        paths['flow'] = cwd + r'/DATA/FPc_00k_80k.h5'
        paths['flow_test'] = cwd + r'/DATA/FPc_00k_03k.h5'
        params['flow']['Re'] = 150
    else:
        paths['flow'] = cwd + r'/DATA/FP_14k_24k.h5'
        paths['flow_test'] = cwd + r'/DATA/FP_10k_13k.h5'
        params['flow']['Re'] = 130


else:
    paths['grid'] = cwd + r'/DATA/SC_grid_AE.mat'
    paths['flow'] = cwd + r'/DATA/SC_00k_00k_AE.mat'
    paths['flow_test'] = cwd + r'/DATA/SC_00k_00k_AE.mat'
    params['flow']['Re'] = 100

# LOGGING
cwd = os.getcwd()
t0 = time.time()
warnings.filterwarnings("ignore")
OUT = {}

log = logging.getLogger()
log.setLevel(logging.INFO)
fh = logging.FileHandler(filename=paths['logger'], mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt='%(asctime)s, %(msecs)d --> %(message)s',
    datefmt='%H:%M:%S'
)
fh.setFormatter(formatter)
log.addHandler(fh)

# LOAD DATA
log.info(f'STARTING loading data')
grid, flow = globals()['read_' + flags['flow']](paths['grid'], paths['flow'])
u = get_control_vector(flow, flags['flow'], flags['control'])

flow_test = globals()['read_' + flags['flow']](paths['grid'], paths['flow_test'])[1]
u_test = get_control_vector(flow_test, flags['flow'], flags['control'])
log.info(f'FINISHED loading data')

# DATA PARAMETERS
params['flow']['n'] = np.shape(grid['X'])[1]  # Columns in grid
params['flow']['m'] = np.shape(grid['X'])[0]  # Rows in grid
log_initial_config(log, params, flags, paths)

# SNAPSHOT MATRIX (TRAIN, VAL, TEST)
D = np.concatenate((flow['U'], flow['V']), axis=0)
Dmean = np.mean(D, axis=1).reshape((np.shape(D)[0]), 1)
Ddt = D - Dmean
t = flow['t']

del D, flow

i_flow = np.linspace(0, np.shape(Ddt)[1] - 1, params['flow']['nt']).astype(int)
Ddt = Ddt[:, i_flow]
t = t[i_flow, :]
if flags['control']:
    u = u[i_flow, :]

Ddt[:,0:3000] = Ddt[:,0:3000] * 130 / 150

D_test = np.concatenate((flow_test['U'], flow_test['V']), axis=0)
Ddt_test = D_test - Dmean
t_test = flow_test['t']
nt_test = len(t_test)
del D_test, flow_test

# GET POD BASIS
if flags['POD']:
    nt = np.shape(Ddt)[1]
    i_snps = random.sample([*range(nt)], params['POD']['nt'])

    nt_POD = params['POD']['nt']
    log.info(f'STARTING POD decomposition for {nt_POD} snapshots')
    Phi, Sigma, Psi = np.linalg.svd(Ddt[:, i_snps], full_matrices=False)
    log.info(f'FINISHED POD decomposition')

    a_test = np.dot(Phi.T, Ddt_test)
    Dr_test_POD = np.dot(Phi[:, 0:params['POD']['nr']], a_test[0:params['POD']['nr'], :])

    cum_energy_POD, energy__POD = energy_POD(Ddt_test, Phi[:, 0:params['POD']['nr']], a_test[0:params['POD']['nr'], :])
    err_POD = get_RMSE(Ddt_test, Dr_test_POD, grid['B'], flags['error_type'])
    CE, CEA = cum_energy_POD['sigma'][-1], cum_energy_POD['acc'][-1]
    log.info(f'Obtained POD energy: RMSE = {err_POD:.2E}, CE = {CE:.2E}, CEA = {CEA:.2E}')

    del CE, CEA, nt_POD

# TRAIN AE
log.info(f'STARTING AE training')
AE = train_AE(params, flags, grid, Ddt, log, u)
del Ddt
log.info(f'FINISHED AE training')
if flags['save']['model']:
    AE.save_weights(paths['model'])
    log.info(f'SAVED AE model')

if flags['get_modal']:

    z_test = get_AE_z(params['AE']['nr'], flags['AE'], AE, grid, Ddt_test)
    log.info(f'TESTING: retrieved latent vector')

    Phi_AE = get_modes_AE(AE, grid, Ddt_test, params['AE']['nr'], flags['AE'], flags['control'], 0, z_test=z_test,
                          b_test=u_test)
    log.info(f'TESTING: retrieved non-static AE modes')
    if not flags['control']:
        Phi_AE_static = get_modes_AE(AE, grid, Ddt_test, params['AE']['nr'], flags['AE'], flags['control'], 1,
                                     b_test=u_test)
        log.info(f'TESTING: retrieved static AE modes')

        if (flags['AE'] == 'MD-CNN-AE') or (flags['AE'] == 'CNN-HAE'):
            cum_energy_AE, energy__AE, i_energy_AE = energy_AE(Ddt_test, Phi_AE, flags['AE'], AE)
        elif (flags['AE'] == 'CNN-VAE') or (flags['AE'] == 'C-CNN-AE'):
            cum_energy_AE, energy__AE, i_energy_AE = energy_AE(Ddt_test, z_test, flags['AE'], AE)

        CEA = cum_energy_AE['acc']
        CE = cum_energy_AE['sigma']
        iE = i_energy_AE['sigma']
        iEA = i_energy_AE['acc']
        log.info(f'TESTING: retrieved AE energy ordering: \n CEA = {CEA}, \n CE = {CE}, \n iE = {iE}, \n iEA = {iEA}')
        del CE, CEA, iE, iEA

        detR, Rij = get_correlation_matrix(Phi_AE_static)
        log.info(f'TESTING: retrieved AE static mode independency: detR = {detR[0]:.2E}, \n Rij = {Rij}')
    detR_t, Rij_t = get_correlation_matrix(Phi_AE)
    log.info(f'TESTING: retrieved AE non-static mode independency')

    if flags['save']['out']:
        OUT['z_test'] = z_test
        if not flags['control']:
            OUT['Phi_AE_static'] = Phi_AE_static
        OUT['Rij_t'] = Rij_t
        OUT['detR_t'] = detR_t

        i_test = np.arange(0, 500, 5)
        for r in range(params['AE']['nr']):
            plot_video_snp(grid, Phi_AE[:, i_test, r],
                           cwd + r'/OUTPUT/' + params['AE']['logger'] + '_Phi' + str(r) + '.gif',
                           limits=[-0.5, 0.5], make_axis_visible=[1, 1], show_title=0,
                           show_colorbar=0, flag_flow=flags['flow'], flag_control=flags['control'],
                           u=u_test, t=t_test)

if flags['get_reconstruction']:

    Dr_test_AE = get_AE_reconstruction(params['AE']['nr'], flags['AE'], flags['control'], AE, grid, Ddt_test, u_test, flag_filter=1)
    z_test = get_AE_z(params['AE']['nr'], flags['AE'], AE, grid, Ddt_test)

    CEA = get_CEA(Ddt_test, Dr_test_AE, grid['B'])
    err_AE = get_RMSE(Ddt_test, Dr_test_AE, grid['B'], flags['error_type'])
    Sc = get_cos_similarity(Ddt_test, Dr_test_AE, grid['B'])
    zdetR, zRij = get_latent_correlation_matrix(z_test.numpy())
    log.info(f'Obtained AE reconstruction: CEA = {CEA:.2E}, RMSE = {err_AE:.2E}, Sc = {Sc:.4f}, zdetR = {zdetR:.2E}')

    if flags['save']['out']:
        OUT['Dr_test_AE'] = Dr_test_AE
        OUT['z_test'] = z_test

if flags['get_latent']:
    z_test = get_AE_z(params['AE']['nr'], flags['AE'], AE, grid, Ddt_test)
    if flags['save']['out']:
        OUT['z_test'] = z_test

if flags['save']['out']:
    with h5py.File(paths['output'], 'w') as h5file:
        for key, item in OUT.items():
            h5file.create_dataset(key, data=item)
    log.info(f'SAVED AE outputs')

if flags['save']['history']:
    t1 = time.time()
    sio.savemat(paths['history'], {'loss': AE.history.history['loss'], 'val_loss': AE.history.history['val_loss'],
                                   'val_energy_loss': AE.history.history['val_energy_loss'],
                                   'energy_loss': AE.history.history['energy_loss'],
                                   'CEA': CEA, 'RMSE_AE': err_AE, 'Sc': Sc, 'zdetR': zdetR, 'zRij': zRij,
                                   'Dtime': t1 - t0})

log.removeHandler(fh)