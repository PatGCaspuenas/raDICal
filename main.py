# GPU REQUIREMENTS
import os
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
import pandas as pd

# LOCAL FUNCTIONS
from utils.data.reader import read_json, read_csv, read_flow, read_latent_space, update_user_vars, prepare_snapshot
from utils.data.loader import load_model_AE, load_model_dyn
from utils.data.logger import logger_initialize, log_initial_params
from utils.data.transformer import flow2window
from utils.data.saver import save_flow_latent, save_dictionary, save_model_AE, save_model_DYN

from utils.AEs.train import train_AE
from utils.AEs.outputs import get_AE_z, get_AE_reconstruction
from utils.AEs.modes import get_modes_AE
from utils.AEs.energy import energy_AE

from utils.dynamics.train import train_dyn
from utils.dynamics.outputs import get_predicted_z, get_predicted_flow

from utils.POD.obtain_basis import get_ROM, get_cumenergy, prepare_POD_snapshot
from utils.modelling.errors_flow import get_CEA, get_RMSE, get_cos_similarity
from utils.modelling.errors_z import get_RMSE_z, get_R2factor, get_latent_correlation_matrix, get_max_w_prop

class Runner:

    def __init__(self):

        # Initialize default variables
        self.PATHS, self.FLAGS, self.PARAMS = read_json()

        # Read commands
        with open(self.PATHS["INPUT_COMMAND"],'r') as f:
            COMMANDS = f.readlines()

        # Iterate over list of user-defined variables
        ITERATIONS = pd.read_csv(self.PATHS["USER_INPUTS"])
        N_iterations = len(ITERATIONS)

        for iteration in range(N_iterations):

            # Update input variables
            self.PATHS, self.FLAGS, self.PARAMS = read_csv(self.PATHS, self.FLAGS, self.PARAMS, iteration)
            self.iteration = iteration

            # Initialize output variables
            self.AE = {}
            self.AE.metrics = {}
            self.DYN = {}
            self.DYN.metrics = {}

            # Initialize logger and write parameters
            log, fh = logger_initialize(self.PATHS, iteration)
            log_initial_params(log, self.PATHS, self.FLAGS, self.PARAMS)

            # WRITE OPTIMIZER FLAG TO CONFIG FILE
            cond_lm = (self.FLAGS['DYN']['optimizer'] == 'LM')
            f = open('.\\utils\\dynamics\\config.py', 'w')
            f.write('cond_lm = ' + repr(cond_lm) + '\n')
            f.close()

            # Evaluate command list
            for command in COMMANDS:
                log.info(f'\n\nEvaluating {command}...\n')
                getattr(self, command)(log)
                log.info(f'Completed {command}\n\n')

            log.removeHandler(fh)
            del log, fh

    def TRAIN_AE(self, log):

        self.AE.model = train_AE(self.PARAMS, self.FLAGS, self.grid, self.D_train, log, b=self.b_train)

    def TRAIN_DYN(self, log):

        self.PARAMS['DYN']['w_o'] = 1
        self.DYN.model = train_dyn(self.PARAMS, self.FLAGS, self.z_train, self.t_train, log, b=self.b_train)

    def GET_MODAL_AE(self, log):

        self.AE.Phi_static = get_modes_AE(self.AE.model, self.grid, self.D_test, self.PARAMS['AE']['N_z'], self.FLAGS['AE']['type'],
                                          self.FLAGS['AE']['control'],1, b_test = self.b_test)

        cum_energy, energy, i_energy = energy_AE(self.D_test, self.z_test, self.FLAGS['AE']['type'], self.AE.model)

        self.AE.metrics.CE_modal = cum_energy['sigma']
        self.AE.metrics.CEA_modal = cum_energy['acc']
        self.AE.metrics.iE_modal = i_energy['sigma']
        self.AE.metrics.iEA_modal = i_energy['acc']


    def GET_MODAL_POD(self, log):

        self.POD = get_ROM(prepare_POD_snapshot(self.D_train, self.PARAMS["POD"]["N_t"]),
                           self.FLAGS["POD"]["r_method"], self.PARAMS["POD"]["r_threshold"])
        self.POD.CE = get_cumenergy(self.POD.Sigma)

    def GET_LAT2FLOW(self, log):

        self.DYN.Dr_test, self.DYN.Dyr_test = get_predicted_flow(self.PARAMS, self.FLAGS, self.PATHS, self.DYN.zy_test,
                                                  self.DYN.zyr_test, by=self.DYN.by_test)
        D_test = flow2window(self.D_test, self.t_test, self.DYN.ty)

        w_props = [5, 20, 50, 100, 200]
        for w_prop in w_props:
            CEA = get_CEA(D_test[:,:,:w_prop], self.DYN.Dyr_test[:,:,:w_prop], self.grid['B'])
            RMSE = get_RMSE(D_test[:,:,:w_prop], self.DYN.Dyr_test[:,:,:w_prop], self.grid['B'], self.FLAGS['AE']['error_type'])
            Sc = get_cos_similarity(D_test[:,:,:w_prop], self.DYN.Dyr_test[:,:,:w_prop], self.grid['B'])
            setattr(self.DYN.metrics, 'CEA_' + str(w_prop), CEA)
            setattr(self.DYN.metrics, 'RMSE_' + str(w_prop), RMSE)
            setattr(self.DYN.metrics, 'Sc_' + str(w_prop), Sc)

        w_prop = self.PARAMS["DYN"]["w_p"]
        self.DYN.metrics.CEA_p = get_CEA(D_test[:, :, :w_prop], self.DYN.Dyr_test[:, :, :w_prop], self.grid['B'])
        self.DYN.metrics.RMSE_p = get_RMSE(D_test[:, :, :w_prop], self.DYN.Dyr_test[:, :, :w_prop], self.grid['B'], self.FLAGS['AE']['error_type'])
        self.DYN.metrics.Sc_p = get_cos_similarity(D_test[:, :, :w_prop], self.DYN.Dyr_test[:, :, :w_prop], self.grid['B'])

        w_prop = self.PARAMS["DYN"]["w_prop"]
        self.DYN.metrics.CEA_prop = get_CEA(D_test[:, :, :w_prop], self.DYN.Dyr_test[:, :, :w_prop], self.grid['B'])
        self.DYN.metrics.RMSE_prop = get_RMSE(D_test[:, :, :w_prop], self.DYN.Dyr_test[:, :, :w_prop], self.grid['B'], self.FLAGS['AE']['error_type'])
        self.DYN.metrics.Sc_prop = get_cos_similarity(D_test[:, :, :w_prop], self.DYN.Dyr_test[:, :, :w_prop], self.grid['B'])

    def GET_FLOW2FLOW(self, log):

        self.AE.Dr_test = get_AE_reconstruction(self.PARAMS['AE']['N_z'], self.FLAGS['AE']['type'], self.FLAGS['AE']['control'],
                                             self.AE.model, self.grid, self.D_test, self.b_test)

        self.AE.metrics.CEA = get_CEA(self.D_test, self.AE.Dr_test, self.grid['B'])
        self.AE.metrics.RMSE = get_RMSE(self.D_test, self.AE.Dr_test, self.grid['B'], self.FLAGS["AE"]['error_type'])
        self.AE.metrics.Sc = get_cos_similarity(self.D_test, self.AE.Dr_test, self.grid['B'])

    def GET_FLOW2LAT(self, log):

        self.z_test = get_AE_z(self.PARAMS['AE']['N_z'], self.FLAGS['AE']['type'], self.AE.model, self.grid, self.D_test)
        self.AE.z_test = self.z_test
        self.AE.metrics.detR, self.AE.metrics.meanR, self.AE.metrics.Rij = get_latent_correlation_matrix(self.z_test)

    def GET_LAT2LAT(self, log):

        self.PARAMS['DYN']['w_o'] = 100

        self.DYN.zx_test, self.DYN.zy_test, self.DYN.zyr_test, self.DYN.bx_test, self.DYN.by_test, T = (
            get_predicted_z(self.PARAMS, self.FLAGS, self.DYN, self.z_test, self.t_test, b=self.b_test))
        self.DYN.ty_test, self.DYN.tx_test = T["PW"], T["TDL"]

        self.DYN.metrics.w_prop_90 = get_max_w_prop(self.DYN.zy_test, self.DYN.zyr_test, self.PARAMS["DYN"]["w_p"])
        self.DYN.metrics.RMSE_prop = get_RMSE_z(self.DYN.zy_test, self.DYN.zyr_test, w_prop=self.PARAMS["DYN"]["w_prop"])
        self.DYN.metrics.RMSE_p = get_RMSE_z(self.DYN.zy_test, self.DYN.zyr_test, w_prop=self.PARAMS["DYN"]["w_p"])
        self.DYN.metrics.R2_p = get_R2factor(self.DYN.zy_test, self.DYN.zyr_test, self.FLAGS["DYN"]["R2_method"],
                                             w_prop=self.PARAMS["DYN"]["w_p"])
        self.DYN.metrics.R2_prop = get_R2factor(self.DYN.zy_test, self.DYN.zyr_test, self.FLAGS["DYN"]["R2_method"],
                                                w_prop=self.PARAMS["DYN"]["w_prop"])

        w_props = [5, 20, 50, 100, 200]
        for w_prop in w_props:
            RMSE = get_RMSE_z(self.DYN.zy_test, self.DYN.zyr_test, w_prop=w_prop)
            R2 = get_R2factor(self.DYN.zy_test, self.DYN.zyr_test, self.FLAGS["DYN"]["R2_method"], w_prop=w_prop)
            setattr(self.DYN.metrics, 'RMSEz_' + str(w_prop), RMSE)
            setattr(self.DYN.metrics, 'R2_' + str(w_prop), R2)

    def LOAD_FLOW_TRAIN(self, log):

        self.grid, flow = read_flow(self.PATHS["GRID"], self.PATHS["FLOW_TRAIN"])
        self.FLAGS, self.PARAMS = update_user_vars(self.PATHS["FLOW_TRAIN"], self.FLAGS, self.PARAMS, grid=self.grid, flow=flow)
        self.D_train, self.t_train, self.b_train = prepare_snapshot(flow, self.PATHS["FLOW_MEAN"], self.FLAGS["FLOW"]["control"],
                                                                      self.FLAGS["FLOW"]["type"], self.PARAMS["FLOW"]["N_t"])
    def LOAD_FLOW_TEST(self, log):

        self.grid, flow = read_flow(self.PATHS["GRID"], self.PATHS["FLOW_TEST"])
        self.FLAGS, self.PARAMS = update_user_vars(self.PATHS["FLOW_TEST"], self.FLAGS, self.PARAMS, grid=self.grid,flow=flow)
        self.D_test, self.t_test, self.b_test = prepare_snapshot(flow, self.PATHS["FLOW_MEAN"],
                                                                      self.FLAGS["FLOW"]["control"],
                                                                      self.FLAGS["FLOW"]["type"],
                                                                      self.PARAMS["FLOW"]["N_t"])
    def LOAD_AE(self, log):

        self.AE.model = load_model_AE(self.PARAMS, self.FLAGS, self.PATHS)

    def LOAD_DYN(self, log):

        self.DYN.model = load_model_dyn(self.PARAMS, self.FLAGS, self.PATHS)

    def LOAD_LATENT_TRAIN(self, log):

        latent = read_latent_space(self.PATHS["LATENT_TRAIN"])
        self.z_train, self.t_train, self.b_train = latent['Z'], latent['t'], latent['U']
        self.FLAGS, self.PARAMS = update_user_vars(self.PATHS["LATENT_TRAIN"], self.FLAGS, self.PARAMS)

    def LOAD_LATENT_TEST(self, log):

        latent = read_latent_space(self.PATHS["LATENT_TEST"])
        self.z_test, self.t_test, self.b_test = latent['Z'], latent['t'], latent['U']
        self.FLAGS, self.PARAMS = update_user_vars(self.PATHS["LATENT_TEST"], self.FLAGS, self.PARAMS)

    def SAVE_AE_MODEL(self, log):

        save_model_AE(self.PATHS["OUTPUTS"], self.iteration, self.AE)

    def SAVE_AE_METRICS(self, log):

        save_dictionary(self.PATHS["OUTPUTS"], self.iteration, self.AE.metrics, 'AE_METRICS')

    def SAVE_AE_FLOW(self, log):

        save_flow_latent(self.PATHS["OUTPUTS"], self.iteration, {'Dr': self.AE.Dr_test, 't': self.t_test},
                         'AE', 'FLOW')

    def SAVE_AE_LATENT(self, log):

        save_flow_latent(self.PATHS["OUTPUTS"], self.iteration, {'z': self.AE.z_test, 't': self.t_test},
                         'AE', 'LATENT')

    def SAVE_DYN_MODEL(self, log):

        save_model_DYN(self.PATHS["OUTPUTS"], self.iteration, self.DYN, self.FLAGS["DYN"]["type"])

    def SAVE_DYN_METRICS(self, log):

        save_dictionary(self.PATHS["OUTPUTS"], self.iteration, self.DYN.metrics, 'DYN_METRICS')

    def SAVE_DYN_FLOW(self, log):

        save_flow_latent(self.PATHS["OUTPUTS"], self.iteration, {'Dr': self.DYN.Dr_test, 'Dyr': self.DYN.Dyr_test,
                         'ty': self.DYN.ty_test, 'tx': self.DYN.tx_test}, 'DYN', 'FLOW')

    def SAVE_DYN_LATENT(self, log):

        save_flow_latent(self.PATHS["OUTPUTS"], self.iteration, {'zx': self.DYN.zx_test, 'zy': self.DYN.zy_test,
                        'zyr': self.DYN.zyr_test, 'bx': self.DYN.bx_test, 'by': self.DYN.by_test,
                        'ty': self.DYN.ty_test, 'tx': self.DYN.tx_test}, 'DYN', 'LATENT')

    def SAVE_POD(self, log):

        save_dictionary(self.PATHS["OUTPUTS"], self.iteration, self.POD, 'POD')

    def DEL_FLOW_TRAIN(self, log):
        del self.D_train, self.t_train, self.b_train
    def DEL_FLOW_TEST(self, log):
        del self.D_test, self.t_test, self.b_test
    def DEL_AE(self, log):
        del self.AE
        self.AE = {}
        self.AE.metrics = {}
    def DEL_DYN(self, log):
        del self.DYN
        self.DYN = {}
        self.DYN.metrics = {}
    def DEL_LATENT_TRAIN(self, log):
        del self.z_train, self.t_train, self.b_train
    def DEL_LATENT_TEST(self, log):
        del self.z_test, self.t_test, self.b_test

if __name__ == 'main':

    runner = Runner()

