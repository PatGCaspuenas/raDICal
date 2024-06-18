# GPU REQUIREMENTS
import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0,2,3'
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

def ROM(params, flags, paths):

    # LOGGING
    t0 = time.time()
    warnings.filterwarnings("ignore")
    OUT = {}
    # logging.basicConfig(filename=paths['logger'],
    #                     filemode='w',
    #                     format='%(asctime)s, %(msecs)d --> %(message)s',
    #                     datefmt='%H:%M:%S',
    #                     level=logging.INFO)
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

    # ORDERING
    cwd = os.getcwd()
    if paths['flow'] == (cwd + r'/DATA/FPc_00k_80k.h5'):
        i_order = np.concatenate((np.arange(6983,10000), np.arange(0,6983)))

    # LOAD DATA
    log.info(f'STARTING loading data')
    grid, flow = globals()['read_' + flags['flow']](paths['grid'], paths['flow'])
    if paths['flow'] == (cwd + r'/DATA/FPc_00k_80k.h5'):
        flow['U'] = flow['U'][:, i_order]
        flow['V'] = flow['V'][:, i_order]
        flow['vF'] = flow['vF'][i_order, :]
        flow['vT'] = flow['vT'][i_order, :]
        flow['vB'] = flow['vB'][i_order, :]
        flow['t'] = flow['t'][i_order, :]
    u = get_control_vector(flow, flags['flow'], flags['control'])

    flow_test = globals()['read_' + flags['flow']](paths['grid'], paths['flow_test'])[1]
    u_test  = get_control_vector(flow_test, flags['flow'], flags['control'])
    log.info(f'FINISHED loading data')

    # DATA PARAMETERS
    params['flow']['n'] = np.shape(grid['X'])[1]  # Columns in grid
    params['flow']['m'] = np.shape(grid['X'])[0]  # Rows in grid
    log_initial_config(log, params,flags,paths)

    # SNAPSHOT MATRIX (TRAIN, VAL, TEST)
    D = np.concatenate( (flow['U'], flow['V']), axis=0)
    Dmean = np.mean(D, axis=1).reshape((np.shape(D)[0]),1)
    Ddt = D - Dmean
    t = flow['t']
    del D, flow

    i_flow = np.linspace(0, np.shape(Ddt)[1]-1, params['flow']['nt']).astype(int)
    Ddt = Ddt[:, i_flow]
    t = t[i_flow, :]
    if flags['control']:
        u = u[i_flow, :]

    D_test = np.concatenate( (flow_test['U'], flow_test['V']), axis=0)
    Ddt_test = D_test - Dmean
    t_test = flow_test['t']
    del D_test, flow_test

    # GET POD BASIS
    if flags['POD']:
        nt = np.shape(Ddt)[1]
        i_snps = random.sample([*range(nt)],params['POD']['nt'])

        nt_POD = params['POD']['nt']
        log.info(f'STARTING POD decomposition for {nt_POD} snapshots')
        Phi, Sigma, Psi = np.linalg.svd(Ddt[:,i_snps], full_matrices=False)
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
        AE.save(paths['model'])
        log.info(f'SAVED AE model')

    if flags['get_modal']:

        z_test = get_AE_z(params['AE']['nr'], flags['AE'], AE, grid, Ddt_test)
        log.info(f'TESTING: retrieved latent vector')

        Phi_AE = get_modes_AE(AE, grid, Ddt_test, params['AE']['nr'], flags['AE'], 0, z_test=z_test)
        log.info(f'TESTING: retrieved non-static AE modes')
        Phi_AE_static = get_modes_AE(AE, grid, Ddt_test, params['AE']['nr'], flags['AE'], 1)
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
            OUT['Phi_AE'] = Phi_AE
            OUT['Phi_AE_static'] = Phi_AE_static
            OUT['Rij_t'] = Rij_t
            OUT['detR_t'] = detR_t


    if flags['get_reconstruction']:

        Dr_test_AE = get_AE_reconstruction(params['AE']['nr'], flags['AE'], flags['control'], AE, grid, Ddt_test, u_test)
        z_test = get_AE_z(params['AE']['nr'], flags['AE'], AE, grid, Ddt_test)

        CEA = get_CEA(Ddt_test, Dr_test_AE, grid['B'])
        err_AE = get_RMSE(Ddt_test, Dr_test_AE, grid['B'], flags['error_type'])
        Sc = get_cos_similarity(Ddt_test, Dr_test_AE, grid['B'])
        zdetR, zRij = get_latent_correlation_matrix(z_test.numpy())
        log.info(f'Obtained AE reconstruction: CEA = {CEA:.2E}, RMSE = {err_AE:.2E}, Sc = {Sc:.4f}, zdetR = {zdetR:.2E}')

        if flags['save']['out']:
            OUT['Dr_test_AE'] = Dr_test_AE

    if flags['save']['out']:
        with h5py.File(paths['output'], 'w') as h5file:
            for key, item in OUT.items():
                h5file.create_dataset(key, data=item)
        log.info(f'SAVED AE outputs')

    if flags['save']['history']:
        t1 = time.time()
        sio.savemat(paths['history'],{'loss': AE.history.history['loss'], 'val_loss': AE.history.history['val_loss'],
                                      'val_energy_loss': AE.history.history['val_energy_loss'],
                                      'energy_loss': AE.history.history['energy_loss'],
                                      'CEA': CEA, 'RMSE_AE': err_AE, 'Sc':Sc, 'zdetR':zdetR, 'zRij': zRij, 'Dtime':t1-t0})

    log.removeHandler(fh)
    del log, fh

