# PACKAGES
import numpy as np
import h5py
import json
import pandas as pd
import re
import tensorflow as tf

# LOAD FUNCTIONS
from utils.data.transformer import get_control_vector
def read_csv(PATHS, FLAGS, PARAMS, i):

    """
    Reads input list with variables selected from the user that are not to be maintained at default values from json files.

    :param PATHS: dictionary with default paths
    :param FLAGS: dictionary with default flags
    :param PARAMS: dictionary with default parameters
    :param i: index of list that is to be evaluated for this iteration of the code
    :return: updated PATHS, FLAGS and PARAMS
    """

    # Read list
    USER_INPUTS = pd.read_csv(PATHS["INPUT_READER"])

    # Get variables from list
    var_names = list(USER_INPUTS.columns.values)

    for var_name in var_names:

        flag_changed = 0
        # To differentiate levels in the dictionaries of FLAGS or PARAMS (e.g. FLOW - type), the % will be used
        var_levels = re.split('%',var_name) # At most two levels

        # This is hard-coded but can be changed if needed
        if (len(var_levels) == 1) and (var_levels[0] in PATHS):
            PATHS[var_levels[0]] = USER_INPUTS[var_name][i]
            flag_changed = 1

        else:
            if var_levels[1] in FLAGS[var_levels[0]]:
                FLAGS[var_levels[0]][var_levels[1]] = USER_INPUTS[var_name][i]
                flag_changed = 1

            elif var_levels[1] in PARAMS[var_levels[0]]:
                PARAMS[var_levels[0]][var_levels[1]] = USER_INPUTS[var_name][i]
                flag_changed = 1

        if not flag_changed:
            raise KeyError("The variable " + var_name + "input is not part of any dictionary")

    # Check if learning rate of AE is adaptive or not and change accordingly
    if FLAGS["AE"]["adaptive_l_r"]:
        step = tf.Variable(0, trainable=False)
        boundaries = [500, 100, 100]
        values = [1e-3, 1e-4, 1e-5, 1e-6]
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        PARAMS['AE']['l_r'] = lr(step)
        del step, boundaries, values

    return PATHS, FLAGS, PARAMS

def read_json():

    """
    Reads default variables from several json files. Their PATH is hard-coded.

    :return: PATHS, FLAGS and PARAMS dictionaries containing all user-defined variables required for the code to run
    """

    PATHS = json.load('.\\INPUTS\\PATHS.json')
    FLAGS = json.load('.\\INPUTS\\FLAGS.json')
    PARAMS = json.load('.\\INPUTS\\PARAMS.json')

    return PATHS, FLAGS, PARAMS

def read_flow(path_grid, path_flow):

    """
    Loads flow and grid .h5 data into different dictionaries. Flow contains velocity components in snapshot matrix form
    (rows spatial points, columns time instants), time and Re; grid contains X, Y and mask B matrices.

    :param path_grid: relative path of grid
    :param path_flow: relative path of flow dataset
    :return: grid and flow dictionaries
    """

    grid = {}
    with h5py.File(path_grid, 'r') as f:
        for i in f.keys():
            grid[i] = f[i][()]

    flow = {}
    with h5py.File(path_flow, 'r') as f:
        for i in f.keys():
            flow[i] = f[i][()]

    return grid, flow

def read_latent_space(path_latent):
    """
    Reads latent space data into dictionary
    :param path_latent: path of latent dataset
    :param flag_control: 1 if control is included in flow, 0 otherwise
    :return: dictionary containing latent space, time array and control vector
    """

    latent = {}
    with h5py.File(path_latent, 'r') as f:
        for i in f.keys():
            latent[i] = f[i][()]

    flag_control = 1 if 'U' in latent else 0
    latent['Z'], latent['t'] = latent['Z'][2:, :], latent['t'][2:, :]

    if flag_control:
        latent['U'] = latent['U'][2:, :]
    else:
        latent['U'] = 0

    return latent

def update_user_vars(path_flow, FLAGS, PARAMS, grid=[], flow=[]):
    """
    Include in FLAGS and PARAMS dictionaries variables about flow dataset

    :param path_flow: relative path of flow dataset or latent space
    :param grid: dictionary containing X, Y and body mask grids
    :param flow: dictionary containing velocity snapshots, time and Re
    :param FLAGS: dictionary with flags
    :param PARAMS: dictionary with parameters
    :return:
    """

    FLAGS["FLOW"]["type"] = path_flow[0:2]
    FLAGS["FLOW"]["control"] = 1 if path_flow[2]=='c' else 0

    if flow:
        PARAMS["FLOW"]["N_y"], PARAMS["FLOW"]["N_x"] = np.shape(grid)

        N_v = np.shape(flow['U'])[0]
        PARAMS["FLOW"]["K"] = N_v // (PARAMS["FLOW"]["N_y"]*PARAMS["FLOW"]["N_x"])

    if (FLAGS["FLOW"]["type"]=='FP') and (FLAGS["FLOW"]["control"]):
        FLAGS["FLOW"]["N_c"] = 3
    else:
        FLAGS["FLOW"]["N_c"] = 0

    return FLAGS, PARAMS

def prepare_snapshot(flow, path_mean, flag_control, flag_type, N_t):
    """
    Prepares snapshot matrix for runner
    :param flow: dictionary containing velocity snapshots, time array and Re
    :param flag_control: 1 if control is included in flow, 0 otherwise
    :param flag_type: type of flow
    :param path_mean: path of mean flow
    :param N_t: number of snapshots to use from snapshot matrix
    :return: snapshot matrix, time array and control vector
    """

    Dmean = np.load(path_mean)

    D = np.concatenate((flow['U'], flow['V']), axis=0)
    Ddt = D - Dmean
    t = flow['t']

    b = get_control_vector(flow, flag_type, flag_control)

    i_flow = np.linspace(0, np.shape(Ddt)[1] - 1, N_t).astype(int)
    Ddt = Ddt[:, i_flow]
    t = t[i_flow, :]
    if flag_control:
        b = b[i_flow, :]

    return Ddt, t, b

