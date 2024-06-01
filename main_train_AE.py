# GPU REQUIREMENTS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 2, 3'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='0'
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

# LOCAL FILES
from utils.data.config import log_initial_config
from utils.data.read_data import read_FP, read_SC
from utils.data.transform_data import get_control_vector

from utils.AEs.modes import get_modes_AE, get_correlation_matrix
from utils.AEs.energy import energy_AE, energy_POD
from utils.AEs.train import train_AE
from utils.AEs.outputs import get_AE_z, get_AE_reconstruction
from utils.modelling.errors import get_RMSE, get_CEA

from utils.plt.plt_snps import *
from utils.modelling.differentiation import get_2Dvorticity

# FLAGS
warnings.filterwarnings("ignore")
flags = {}
flags['save'] = {}

flags['loss'] = 'energy'        # Type of loss metric (mse, energy)
flags['AE'] = 'C-CNN-AE'         # AE type (C-CNN-AE, MD-CNN-AE, CNN-HAE, CNN-VAE)
flags['struct'] = 'simple'     # AE structure type (simple, medium, complex)
flags['flow'] = 'FP'            # Flow type (SC, FP)
flags['control'] = 1            # With (1) or without (0) control
flags['POD'] = 0                # Gets POD basis (1) or not (0)
flags['lr_static'] = 0          # Fixed learning rate (1) or varying with nepochs (0)
flags['get_modal'] = 0          # Retrieve modal analysis
flags['get_reconstruction'] = 1 # Retrieve AE reconstruction and error quantification
flags['save']['model'] = 0      # Save trained model (1) or not (0)
flags['save']['out'] = 0        # Save outputs of AE (1) or not (0)
flags['save']['history'] = 1    # Save model history loss (1) or not (0)
flags['error_type'] = 'W'       # Type of RMSE

# PARAMETERS
params = {}
params['AE'] = {}
params['flow'] = {}
params['POD'] = {}

# Learning rate in Adam optimizer
if flags['lr_static']:
    params['AE']['lr'] = 1e-4
else:
    step = tf.Variable(0, trainable=False)
    boundaries = [500, 100, 100]
    values = [1e-3, 1e-4, 1e-5, 1e-6]
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    params['AE']['lr'] = lr
    del step, boundaries, values

params['AE']['logger'] = 'test'          # Logger name
params['AE']['n_epochs'] = 10             # Number of epochs
params['AE']['batch_size'] = 32          # Batch size when training AE
params['AE']['beta'] = 1e-3              # CNN-VAE reg parameter
params['AE']['ksize'] = (3,3)            # Kernel size of convolutional layers
params['AE']['psize'] = (2,2)            # Kernel size of pooling layers
params['AE']['ptypepool'] = 'valid'      # Type of pooling padding (same or valid)
params['AE']['nstrides'] = 2             # Number of strides in pooling
params['AE']['act'] = 'tanh'             # Activation function
params['AE']['reg_k'] = 0                # Regularization kernel
params['AE']['reg_b'] = 0                # Regularization bias
params['AE']['nr'] = 2                   # AE latent space dimensions

params['flow']['k'] = 2                  # Number of dimensions

params['POD']['nt'] = 50                 # Number of snapshots to create POD basis
params['POD']['nr'] = params['AE']['nr'] # Number of modes in truncated basis

# PATHS
paths = {}
cwd = os.getcwd()

paths['history'] = cwd + r'/OUTPUT/' + params['AE']['logger'] + '_history.mat'
paths['output'] = cwd + r'/OUTPUT/' + params['AE']['logger'] + '_out.h5'
paths['model'] = cwd + r'/MODELS/' + params['AE']['logger'] + '_model.keras'
paths['logger'] = cwd + r'/OUTPUT/' + params['AE']['logger'] + '_logger.log'

if flags['flow']=='FP':
    paths['grid'] = cwd + r'/DATA/FP_grid.h5'

    if flags['control']:
        paths['flow'] = cwd + r'/DATA/FPc_00k_80k.h5'
        paths['flow_test'] = cwd + r'/DATA/FPc_00k_03k.h5'
        params['flow']['Re'] = 130
    else:
        paths['flow'] = cwd + r'/DATA/FP_10k_13k.h5'
        paths['flow_test'] = cwd + r'/DATA/FP_10k_13k.h5'
        params['flow']['Re'] = 150


else:
    paths['grid'] = cwd + r'/DATA/SC_grid_AE.mat'
    paths['flow'] = cwd + r'/DATA/SC_00k_00k_AE.mat'
    paths['flow_test'] = cwd + r'/DATA/SC_00k_00k_AE.mat'
    params['flow']['Re'] = 100

# LOGGING
OUT = {}
logging.basicConfig(filename=paths['logger'],
                    filemode='w',
                    format='%(asctime)s, %(msecs)d --> %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

# LOAD DATA
logging.info(f'STARTING loading data')
grid, flow = locals()['read_' + flags['flow']](paths['grid'], paths['flow'])
u = get_control_vector(flow, flags['flow'], flags['control'])

flow_test = locals()['read_' + flags['flow']](paths['grid'], paths['flow_test'])[1]
u_test  = get_control_vector(flow_test, flags['flow'], flags['control'])
logging.info(f'FINISHED loading data')

# DATA PARAMETERS
params['flow']['n'] = np.shape(grid['X'])[1]  # Columns in grid
params['flow']['m'] = np.shape(grid['X'])[0]  # Rows in grid
log_initial_config(logging, params,flags,paths)

# SNAPSHOT MATRIX (TRAIN, VAL, TEST)
D = np.concatenate( (flow['U'], flow['V']), axis=0)
Dmean = np.mean(D, axis=1).reshape((np.shape(D)[0]),1)
Ddt = D - Dmean
t = flow['t']
del D, flow

D_test = np.concatenate( (flow_test['U'], flow_test['V']), axis=0)
Ddt_test = D_test - Dmean
t_test = flow_test['t']
del D_test, flow_test

# GET POD BASIS
if flags['POD']:
    nt = np.shape(Ddt)[1]
    i_snps = random.sample([*range(nt)],params['POD']['nt'])

    nt_POD = params['POD']['nt']
    logging.info(f'STARTING POD decomposition for {nt_POD} snapshots')
    Phi, Sigma, Psi = np.linalg.svd(Ddt[:,i_snps], full_matrices=False)
    logging.info(f'FINISHED POD decomposition')

    a_test = np.dot(Phi.T, Ddt_test)
    Dr_test_POD = np.dot(Phi[:, 0:params['POD']['nr']], a_test[0:params['POD']['nr'], :])

    cum_energy_POD, energy_POD = energy_POD(Ddt_test, Phi[:, 0:params['POD']['nr']], a_test[0:params['POD']['nr'], :])
    err_POD = get_RMSE(Ddt_test, Dr_test_POD, grid['B'], flags['error_type'])
    CE, CEA = cum_energy_POD['sigma'][-1], cum_energy_POD['acc'][-1]
    logging.info(f'Obtained POD energy: RMSE = {err_POD:.2E}, CE = {CE:.2E}, CEA = {CEA:.2E}')

    del CE, CEA, nt_POD

# TRAIN AE
logging.info(f'STARTING AE training')
AE = train_AE(params, flags, grid, Ddt, logging, u)
del Ddt
logging.info(f'FINISHED AE training')
if flags['save']['model']:
    AE.save(paths['model'])
    logging.info(f'SAVED AE model')

if flags['get_modal']:

    z_test = get_AE_z(params['AE']['nr'], flags['AE'], AE, grid, Ddt_test)
    logging.info(f'TESTING: retrieved latent vector')

    Phi_AE = get_modes_AE(AE, grid, Ddt_test, params['AE']['nr'], flags['AE'], 0, z_test=z_test)
    logging.info(f'TESTING: retrieved non-static AE modes')
    Phi_AE_static = get_modes_AE(AE, grid, Ddt_test, params['AE']['nr'], flags['AE'], 1)
    logging.info(f'TESTING: retrieved static AE modes')

    if (flags['AE'] == 'MD-CNN-AE') or (flags['AE'] == 'CNN-HAE'):
        cum_energy_AE, energy_AE, i_energy_AE = energy_AE(Ddt_test, Phi_AE, flags['AE'], AE)
    elif (flags['AE'] == 'CNN-VAE') or (flags['AE'] == 'C-CNN-AE'):
        cum_energy_AE, energy_AE, i_energy_AE = energy_AE(Ddt_test, z_test, flags['AE'], AE)

    CEA = cum_energy_AE['acc']
    CE = cum_energy_AE['sigma']
    iE = i_energy_AE['sigma']
    iEA = i_energy_AE['acc']
    logging.info(f'TESTING: retrieved AE energy ordering: \n CEA = {CEA}, \n CE = {CE}, \n iE = {iE}, \n iEA = {iEA}')
    del CE, CEA, iE, iEA

    detR, Rij = get_correlation_matrix(Phi_AE_static)
    logging.info(f'TESTING: retrieved AE static mode independency: detR = {detR[0]:.2E}, \n Rij = {Rij}')
    detR_t, Rij_t = get_correlation_matrix(Phi_AE)
    logging.info(f'TESTING: retrieved AE non-static mode independency')

    if flags['save']['out']:
        OUT['z_test'] = z_test
        OUT['Phi_AE'] = Phi_AE
        OUT['Phi_AE_static'] = Phi_AE_static
        OUT['Rij_t'] = Rij_t
        OUT['detR_t'] = detR_t


if flags['get_reconstruction']:

    Dr_test_AE = get_AE_reconstruction(params['AE']['nr'], flags['AE'], flags['control'], AE, grid, Ddt_test, u_test)
    CEA = get_CEA(Ddt_test, Dr_test_AE, grid['B'])
    err_AE = get_RMSE(Ddt_test, Dr_test_AE, grid['B'], flags['error_type'])
    logging.info(f'Obtained AE reconstruction: CEA = {CEA:.2E}, RMSE = {err_AE:.2E}')

    if flags['save']['out']:
        OUT['Dr_test_AE'] = Dr_test_AE

if flags['save']['out']:
    with h5py.File(paths['output'], 'w') as h5file:
        for key, item in OUT.items():
            h5file.create_dataset(key, data=item)
    logging.info(f'SAVED AE outputs')

if flags['save']['history']:
    sio.savemat(paths['history'],{'loss': AE.history.history['loss'], 'val_loss': AE.history.history['val_loss']})


