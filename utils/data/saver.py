# PACKAGES
import scipy.io as sio
import pickle
import os
import h5py

def save_dictionary(path_out, it, OUT, flag_name):
    """
    Saves any dictionary in .mat format
    :param path_out: output folder path
    :param it: iteration number in csv list
    :param OUT: dictionary
    :param flag_name: name to save dictionary
    """

    path_save = os.path.join(path_out, flag_name + '_' + str(it) + '.mat')
    sio.savemat(path_save, OUT)

def save_model_AE(path_out, it, AE):
    """
    Saves AE model with pickle
    :param path_out: output folder path
    :param it: iteration number in csv list
    :param AE: AE model class
    """

    path_encoder = os.path.join(path_out, 'MODEL_AE_enc_' + str(it))
    path_decoder = os.path.join(path_out, 'MODEL_AE_dec_' + str(it))

    with open(path_encoder, "wb") as fp:  # Pickling
        pickle.dump(AE.encoder.get_weights(), fp)
    with open(path_decoder, "wb") as fp:  # Pickling
        pickle.dump(AE.decoder.get_weights(), fp)

def save_model_DYN(path_out, it, DYN, flag_DYN):
    """
    Saves dynamica predictor model with pickle
    :param path_out: output folder path
    :param it: iteration number in csv list
    :param DYN: dynamical predictor model class
    :param flag_DYN: type of dynamical predictor (LSTM or NARX)
    """

    if flag_DYN == 'LSTM':
        path_a = os.path.join(path_out, 'MODEL_DYN_LSTMa_' + str(it))
        path_b = os.path.join(path_out, 'MODEL_DYN_LSTMb_' + str(it))
        with open(path_a, "wb") as fp:  # Pickling
            pickle.dump(DYN.LSTMs.get_weights(), fp)
        with open(path_b, "wb") as fp:  # Pickling
            pickle.dump(DYN.MLPs.get_weights(), fp)

    else:
        path_save = os.path.join(path_out, 'MODEL_DYN_NARX_' + str(it))
        with open(path_save, "wb") as fp:  # Pickling
            pickle.dump(DYN.MLP.get_weights(), fp)

def save_flow_latent(path_out, it, OUT, flag_AE_DYN, flag_flow_lat):
    """
    Saves any dictionary in .h5 format
    :param path_out: output folder path
    :param it: iteration number in csv list
    :param OUT: dictionary
    :param flag_AE_DYN: AE or DYN flag
    :param flag_flow_lat: FLOW or LATENT flag
    """

    path_save = os.path.join(path_out, flag_flow_lat + '_' + flag_AE_DYN + '_' + str(it) + '.h5')

    with h5py.File(path_save, 'w') as h5file:
        for key, item in OUT.items():
            h5file.create_dataset(key, data=item)

