# GPU REQUIREMENTS
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
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
import os
import ast
import numpy as np
import pandas as pd
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
from utils.data.transform_data import get_control_vector, flow2window

from utils.dynamics.outputs import get_predicted_z, get_predicted_flow
from utils.dynamics.train import train_dyn
from utils.modelling.errors_flow import get_RMSE, get_CEA, get_cos_similarity
from utils.modelling.errors_z import get_RMSE_z, get_max_ntpred, get_R2factor

# ITERABLES
cwd = os.getcwd()
IT = pd.read_csv(cwd+r'/OUTPUT/dyn_1st_control.csv')

for i in range(len(IT)):
    # FLAGS
    flags = {}
    flags['save'] = {}
    flags['dyn'] = {}

    flags['AE'] = IT['AE'][i]          # AE type (C-CNN-AE, MD-CNN-AE, CNN-HAE, CNN-VAE)
    flags['struct'] = 'complex'     # AE structure type (simple, medium, complex)
    flags['flow'] = 'FP'            # Flow type (SC, FP)
    flags['POD'] = 0
    flags['get_modal'] = 0
    flags['get_reconstruction'] = 0
    flags['lr_static'] = 1
    flags['control'] = 1            # With (1) or without (0) control
    flags['filter'] = 0

    flags['decode'] = 1             # Decode from trained model the predicted latent space and error quantification
    flags['save']['model'] = 0      # Save trained model (1) or not (0)
    flags['save']['out'] = 1
    flags['save']['latent'] = 1      # Save outputs of AE (1) or not (0)
    flags['save']['history'] = 1    # Save model history loss (1) or not (0)
    flags['error_type'] = 'W'       # Type of RMSE
    flags['R2_type'] = 'C'          # Type of R2 coefficient (correlation of determination)

    flags['dyn']['control'] = int(IT['flag_control_dyn'][i])
    flags['dyn']['type'] = (IT['dyn'][i])
    flags['dyn']['multi_train'] = 0
    flags['dyn']['opt'] = (IT['opt'][i])

    # PARAMETERS
    params = {}
    params['AE'] = {}
    params['flow'] = {}
    params['POD'] = {}
    params['dyn'] = {}
    params['LSTM'] = {}
    params['NARX'] = {}

    params['POD']['nt'] = 1000
    params['POD']['nr'] = 100

    # Learning rate in Adam optimizer
    params['AE']['logger'] = IT['AE'][i] + '_lr_' + str(IT['lr'][i]) + \
                             '_nepoch_' + str(IT['n_epochs'][i]) + \
                             '_batch_' + str(IT['batch_size'][i]) + \
                             '_beta_' + str(IT['beta'][i] ) + \
                             '_nr_' + str(IT['nr'][i] ) + \
                             '_nt_' + str(IT['nt'][i] ) # Logger name
    params['dyn']['logger'] = IT['dyn'][i] + '_' + IT['AE'][i] + '_lr_' + str(IT['lr_d'][i]) + \
                             '_nepoch_' + str(IT['n_epochs_d'][i]) + \
                             '_batch_' + str(IT['batch_size_d'][i]) + \
                              '_fc_' + str(IT['flag_control_dyn'][i]) + \
                              '_d_' + str(IT['d'][i]) + \
                             '_np_' + str(IT['np'][i]) + \
                             '_lstmu_' + str(IT['lstm_units'][i]) + \
                             '_lstmdu_' + str(IT['dense_units'][i]) + \
                             '_narxdu_' + str(IT['units'][i])  # Logger name

    params['AE']['lr'] = IT['lr'][i]
    params['AE']['n_epochs'] = int(IT['n_epochs'][i])    # Number of epochs
    params['AE']['batch_size'] = int(IT['batch_size'][i]) # Batch size when training AE
    params['AE']['beta'] = IT['beta'][i]             # CNN-VAE reg parameter
    params['AE']['ksize'] = (3,3)                    # Kernel size of convolutional layers
    params['AE']['psize'] = (2,2)                    # Kernel size of pooling layers
    params['AE']['ptypepool'] = 'valid'              # Type of pooling padding (same or valid)
    params['AE']['nstrides'] = 2                     # Number of strides in pooling
    params['AE']['act'] = 'tanh'                     # Activation function
    params['AE']['reg_k'] = 0                        # Regularization kernel
    params['AE']['reg_b'] = 0                        # Regularization bias
    params['AE']['nr'] = int(IT['nr'][i])            # AE latent space dimensions

    params['flow']['k'] = 2                          # Number of dimensions
    params['flow']['nt'] = int(IT['nt'][i])
    params['flow']['nc'] = int(IT['nc'][i]) if flags['dyn']['control'] else 0

    params['dyn']['n_epochs'] = int(IT['n_epochs_d'][i])
    params['dyn']['batch_size'] = int(IT['batch_size_d'][i])
    params['dyn']['lr'] = IT['lr_d'][i]

    params['dyn']['np'] = int(IT['np'][i])
    params['dyn']['d'] = int(IT['d'][i])
    params['dyn']['act'] = 'tanh'
    params['dyn']['nt_pred'] = 100 + params['dyn']['np']

    params['dyn']['dt'] = 0.1
    params['dyn']['dt_lat'] = 0.1
    params['dyn']['irreg'] = 0

    params['dyn']['o'] = 1

    params['dyn']['kreg'] = 0

    params['LSTM']['lstm_units'] = ast.literal_eval((IT['lstm_units'][i]))
    params['LSTM']['dense_units'] = ast.literal_eval((IT['dense_units'][i]))
    params['LSTM']['dropout'] = np.zeros((len(params['LSTM']['dense_units'])))

    params['NARX']['units'] = ast.literal_eval((IT['units'][i]))
    params['NARX']['dropout'] = np.zeros((len(params['NARX']['units'])))

    # PATHS
    paths = {}

    paths['history'] = cwd + r'/OUTPUT/' + params['dyn']['logger'] + '_history.mat'
    paths['output'] = cwd + r'/OUTPUT/' + params['dyn']['logger'] + '_out.h5'
    paths['logger'] = cwd + r'/OUTPUT/' + params['dyn']['logger'] + '_logger.log'

    if flags['flow']=='FP':
        paths['grid'] = cwd + r'/DATA/FP_grid.h5'

        if flags['control']:
            paths['flow'] = cwd + r'/DATA/FPc_00k_70k.h5'
            paths['flow_test'] = cwd + r'/DATA/FPc_00k_03k.h5'

            if flags['AE'] == 'CNN-VAE':
                paths['z'] = cwd + r'/DATA/latent/FPcz_00k_10k_CNNVAE.h5'
                paths['z_test'] = cwd + r'/DATA/latent/FPcz_00k_03k_CNNVAE.h5'
                paths['model'] = r'cCNN-VAE'
            else:
                paths['z'] = cwd + r'/DATA/latent/FPcz_00k_10k_CCNNAE.h5'
                paths['z_test'] = cwd + r'/DATA/latent/FPcz_00k_03k_CCNNAE.h5'
                paths['model'] = r'cC-CNN-AE'

            params['flow']['Re'] = 150
        else:
            paths['flow'] = cwd + r'/DATA/FP_14k_24k.h5'
            paths['flow_test'] = cwd + r'/DATA/FP_10k_13k.h5'

            if flags['AE'] == 'CNN-VAE':
                paths['z'] = cwd + r'/DATA/latent/FPz_14k_24k_CNNVAE.h5'
                paths['z_test'] = cwd + r'/DATA/latent/FPz_10k_13k_CNNVAE.h5'
                paths['model'] = r'CNN-VAE'
            else:
                paths['z'] = cwd + r'/DATA/latent/FPz_14k_24k_CCNNAE.h5'
                paths['z_test'] = cwd + r'/DATA/latent/FPz_10k_13k_CCNNAE.h5'
                paths['model'] = r'C-CNN-AE'

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
    grid, flow_test = globals()['read_' + flags['flow']](paths['grid'], paths['flow_test'])
    u_test  = get_control_vector(flow_test, flags['flow'], flags['control'])
    log.info(f'FINISHED loading data')

    # DATA PARAMETERS
    params['flow']['n'] = np.shape(grid['X'])[1]  # Columns in grid
    params['flow']['m'] = np.shape(grid['X'])[0]  # Rows in grid
    log_initial_config(log, params,flags,paths)

    # SNAPSHOT MATRIX (TRAIN, VAL, TEST)
    if flags['control']:
        Dmean = np.load(cwd + r'/DATA/others/FPc_Dmean.npy')
    else:
        Dmean = np.load(cwd + r'/DATA/others/FP_Dmean.npy')

    D_test = np.concatenate( (flow_test['U'], flow_test['V']), axis=0)
    Ddt_test = D_test - Dmean
    t_test_flow = flow_test['t']
    nt_test = len(t_test_flow)
    del D_test, flow_test

    # LOAD LATENT SPACE (TRAIN, VAL, TEST)
    latent = {}
    latent_test = {}

    with h5py.File(paths['z'], 'r') as f:
        for i in f.keys():
            latent[i] = f[i][()]
    with h5py.File(paths['z_test'], 'r') as f:
        for i in f.keys():
            latent_test[i] = f[i][()]

    if flags['control']:
        Z, t, U = latent['Z'], latent['t'], latent['U']
        Z_test, t_test, U_test = latent_test['Z'], latent_test['t'], latent_test['U']
    else:
        Z, t = latent['Z'], latent['t']
        Z_test, t_test = latent_test['Z'], latent_test['t']

    # TRAIN DYNAMIC PREDICTOR
    t0train = time.time()
    if flags['control']:
        DYN, Znorm = train_dyn(params, flags, Z, t, logging, u=U)
    else:
        DYN, Znorm = train_dyn(params, flags, Z, t, logging)
    t1train = time.time()

    # PREDICT LATENT SPACE AND DECODE
    params['dyn']['o'] = 10
    t0pred = time.time()
    if flags['control']:
        Zx_test, Zy_test, Zy_test_dyn, Ux_test, Uy_test, T = get_predicted_z(params, flags, DYN, Z_test, t_test, Znorm, u=U_test)
        if flags['save']['out']:
            OUT['Zx_test'] = Zx_test
            OUT['Zy_test'] = Zy_test
            OUT['t_test'] = T['pred']
            OUT['Zy_dyn_test'] = Zy_test_dyn
            OUT['Ux_test'] = Ux_test
            OUT['Uy_test'] = Uy_test

    else:
        Zx_test, Zy_test, Zy_test_dyn, T = get_predicted_z(params, flags, DYN, Z_test, t_test, Znorm, u=0)
        if flags['save']['out']:
            OUT['Zx_test'] = Zx_test
            OUT['Zy_test'] = Zy_test
            OUT['t_test'] = T['pred']
            OUT['Zy_dyn_test'] = Zy_test_dyn
    t1pred = time.time()

    err_dyn = get_RMSE_z(Zy_test, Zy_test_dyn, n_p = params['dyn']['np'])
    NT = get_max_ntpred(Zy_test, Zy_test_dyn, params['dyn']['np'])
    R2C = get_R2factor(Zy_test, Zy_test_dyn, 'C', n_p = params['dyn']['np'])
    err_dyn_ntpred = get_RMSE_z(Zy_test, Zy_test_dyn, n_p=params['dyn']['nt_pred'])
    R2C_ntpred = get_R2factor(Zy_test, Zy_test_dyn, 'C', n_p=params['dyn']['nt_pred'])
    log.info(f'Obtained dyn prediction: RMSE = {err_dyn}, NT = {NT}, R2C = {R2C}')

    if flags['decode']:

        if flags['control']:
            Ddtr_test, Ddtr_dyn_test = get_predicted_flow(params, flags, paths, Zy_test, Zy_test_dyn, Uy=Uy_test)
        else:
            Ddtr_test, Ddtr_dyn_test = get_predicted_flow(params, flags, paths, Zy_test, Zy_test_dyn, Uy=0)


        Ddt_test = flow2window(Ddt_test, t_test_flow, T['pred'])
        CEA_np = get_CEA(Ddt_test[:,:,0:params['dyn']['np']], Ddtr_dyn_test[:,:,0:params['dyn']['np']], grid['B'])
        err_AE_np = get_RMSE(Ddt_test[:,:,0:params['dyn']['np']], Ddtr_dyn_test[:,:,0:params['dyn']['np']], grid['B'], flags['error_type'])
        Sc_np = get_cos_similarity(Ddt_test[:,:,0:params['dyn']['np']], Ddtr_dyn_test[:,:,0:params['dyn']['np']], grid['B'])

        CEA_nt = get_CEA(Ddt_test[:,:,0:NT], Ddtr_dyn_test[:,:,0:NT], grid['B'])
        err_AE_nt = get_RMSE(Ddt_test[:,:,0:NT], Ddtr_dyn_test[:,:,0:NT], grid['B'], flags['error_type'])
        Sc_nt = get_cos_similarity(Ddt_test[:,:,0:NT], Ddtr_dyn_test[:,:,0:NT], grid['B'])
        CEA_ntpred = get_CEA(Ddt_test, Ddtr_dyn_test, grid['B'])
        err_AE_ntpred = get_RMSE(Ddt_test, Ddtr_dyn_test, grid['B'], flags['error_type'])
        Sc_ntpred = get_cos_similarity(Ddt_test, Ddtr_dyn_test, grid['B'])

        log.info(f'Obtained AE reconstruction: CEA_np = {CEA_np:.2E}, RMSE_np = {err_AE_np:.2E}, Sc_np = {Sc_np:.4f}, CEA_nt = {CEA_nt:.2E}, RMSE_nt = {err_AE_nt:.2E}, Sc_np = {Sc_nt:.4f}')

        if flags['save']['out'] and not flags['save']['latent']:
            OUT['Ddt_test'] = Ddt_test
            OUT['Ddtr_test'] = Ddtr_test
            OUT['Ddtr_dyn_test'] = Ddtr_dyn_test

    # SAVE ALL OUTPUTS
    if flags['save']['out']:
        with h5py.File(paths['output'], 'w') as h5file:
            for key, item in OUT.items():
                h5file.create_dataset(key, data=item)
        log.info(f'SAVED AE outputs')

    # SAVE HISTORY
    if flags['save']['history']:
        t1 = time.time()
        if flags['dyn']['opt'] == 'Adam':
            loss = DYN.history.history['loss']
        else:
            loss = DYN.history.history['train_loss']

        if flags['decode']:
            sio.savemat(paths['history'],{'loss': loss, 'val_loss': DYN.history.history['val_loss'],
                                          'CEA_np': CEA_np, 'RMSE_AE_np': err_AE_np, 'Sc_np':Sc_np,
                                          'CEA_nt': CEA_nt, 'RMSE_AE_nt': err_AE_nt, 'Sc_nt': Sc_nt,
                                          'CEA_ntpred': CEA_ntpred, 'RMSE_AE_ntpred': err_AE_ntpred, 'Sc_ntpred': Sc_ntpred,
                                          'RMSE_dyn': err_dyn, 'NT': NT, 'R2C':R2C,
                                          'RMSE_dyn_ntpred': err_dyn_ntpred, 'R2C_ntpred': R2C_ntpred,
                                          'Dtime':t1-t0, 'Dt_train':t1train - t0train, 'Dt_pred':t1pred - t0pred})
        else:
            sio.savemat(paths['history'],{'loss': DYN.history.history['loss'], 'val_loss': DYN.history.history['val_loss'],
                                          'RMSE_dyn': err_dyn, 'NT': NT, 'R2C':R2C,
                                          'RMSE_dyn_ntpred': err_dyn_ntpred, 'R2C_ntpred': R2C_ntpred,
                                          'Dtime':t1-t0, 'Dt_train':t1train - t0train, 'Dt_pred':t1pred - t0pred})

        log.removeHandler(fh)
        del log, fh