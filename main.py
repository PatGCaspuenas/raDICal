# PACKAGES
import tensorflow as tf
import os
import pandas as pd

# LOCAL FUNCTIONS
from autoencoder import ROM

# ITERABLES
cwd = os.getcwd()
flags_AE = ['CNN-VAE', 'C-CNN-AE']
IT = pd.read_csv(cwd+r'/OUTPUT/FP_no_control.csv')

for flag_AE in flags_AE:
    for i in range(len(IT)):
        # FLAGS
        flags = {}
        flags['save'] = {}

        flags['AE'] = flag_AE           # AE type (C-CNN-AE, MD-CNN-AE, CNN-HAE, CNN-VAE)
        flags['struct'] = 'complex'     # AE structure type (simple, medium, complex)
        flags['flow'] = 'FP'            # Flow type (SC, FP)
        flags['control'] = 0            # With (1) or without (0) control
        flags['POD'] = 0                # Gets POD basis (1) or not (0)

        if IT['lr'][i] == 0:
            flags['lr_static'] = 0      # Fixed learning rate (1) or varying with nepochs (0)
        else:
            flags['lr_static'] = 1

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
            params['AE']['lr'] = IT['lr'][i]
        else:
            step = tf.Variable(0, trainable=False)
            boundaries = [500, 100, 100]
            values = [1e-3, 1e-4, 1e-5, 1e-6]
            lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
            params['AE']['lr'] = lr
            del step, boundaries, values

        params['AE']['logger'] = flag_AE + '_lr_' + str(IT['lr'][i]) + \
                                 '_nepoch_' + str(IT['n_epochs'][i]) + \
                                 '_batch_' + str(IT['batch_size'][i]) + \
                                 '_beta_' + str(IT['beta'][i] ) + \
                                 '_nr_' + str(IT['nr'][i] ) + \
                                 '_nt_' + str(IT['nt'][i] )   # Logger name

        params['AE']['n_epochs'] = IT['n_epochs'][i]     # Number of epochs
        params['AE']['batch_size'] = IT['batch_size'][i] # Batch size when training AE
        params['AE']['beta'] = IT['beta'][i]             # CNN-VAE reg parameter
        params['AE']['ksize'] = (3,3)                    # Kernel size of convolutional layers
        params['AE']['psize'] = (2,2)                    # Kernel size of pooling layers
        params['AE']['ptypepool'] = 'valid'              # Type of pooling padding (same or valid)
        params['AE']['nstrides'] = 2                     # Number of strides in pooling
        params['AE']['act'] = 'tanh'                     # Activation function
        params['AE']['reg_k'] = 0                        # Regularization kernel
        params['AE']['reg_b'] = 0                        # Regularization bias
        params['AE']['nr'] = IT['nr'][i]                 # AE latent space dimensions

        params['flow']['k'] = 2                          # Number of dimensions
        params['flow']['nt'] = IT['nt'][i]

        params['POD']['nt'] = 500                         # Number of snapshots to create POD basis
        params['POD']['nr'] = params['AE']['nr']         # Number of modes in truncated basis

        # PATHS
        paths = {}

        paths['history'] = cwd + r'/OUTPUT/' + params['AE']['logger'] + '_history.mat'
        paths['output'] = cwd + r'/OUTPUT/' + params['AE']['logger'] + '_out.h5'
        paths['model'] = cwd + r'/MODELS/' + params['AE']['logger'] + '_model.keras'
        paths['logger'] = cwd + r'/OUTPUT/' + params['AE']['logger'] + '_logger.log'

        if flags['flow']=='FP':
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

        ROM(params, flags, paths)