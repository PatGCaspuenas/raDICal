# PACKAGES
import logging
import os
from keras.callbacks import Callback
from timeit import default_timer as timer

class MyLogger(Callback):
    """
    Integration of logging of each epoch on existent logger
    """
    def __init__(self,logging, epochs):
        super(MyLogger, self).__init__()
        self.logging = logging
        self.n_epoch = epochs
    def on_train_batch_end(self, batch, logs=None):
        self.batch_n += 1
    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = timer()
        self.batch_n = 0
    def on_epoch_end(self, epoch, logs=None):
        self.logging.info(f'Epoch {epoch}/{self.n_epoch} - {self.batch_n}/{self.batch_n} - {(timer()-self.starttime)}s - {logs}')
def log_initial_params(log, PATHS, FLAGS, PARAMS):

    """
    Log parameters (default and user input)

    :param log: logger object
    :param PATHS: dictionary with paths
    :param FLAGS: dictionary with flags
    :param PARAMS: dictionary with params
    :return: None
    """

    # PATHS
    log.info('\n\nPATHS\n\n')
    for k in PATHS.keys():
        log.info(f'{k}\t\t : {PATHS[k]}\n')

    # FLAGS
    log.info('\n\nFLAGS\n\n')
    for k1 in FLAGS.keys():
        for k2 in FLAGS[k1].keys():
            log.info(f'{k1}[{k2}]\t\t : {FLAGS[k1][k2]}\n')

    # PARAMS
    log.info('\n\nPARAMS\n\n')
    for k1 in PARAMS.keys():
        for k2 in PARAMS[k1].keys():
            log.info(f'{k1}[{k2}]\t\t : {PARAMS[k1][k2]}\n')

def logger_initialize(PATHS, i):
    """
    Create logger

    :param PATHS: dictionary with paths
    :param i: iteration index in user defined list
    :return: logger and handler object
    """

    path_logger = os.path.join(PATHS["OUTPUTS"], 'LOGGER_' + str(i+1) + '.log')

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=path_logger, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
                    fmt='%(asctime)s, %(msecs)d --> %(message)s',
                    datefmt='%H:%M:%S'
                    )
    fh.setFormatter(formatter)
    log.addHandler(fh)

    return log, fh