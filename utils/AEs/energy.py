# PACKAGES
import numpy as np
import tensorflow as tf

# LOCAL FUNCTIONS
from utils.data.transformer import CNNAE2raw

def energy_POD(D, Phi, a):
    """
    Retrieves energy and cumulative energy up to x number of modes (from 1 to N_r)

    :param D: snapshot matrix (N_v, N_t)
    :param Phi: spatial mode set (N_v, N_r)
    :param a: temporal mode set (N_r, N_t)
    :return: cumulative energy and energy arrays
    """

    # NUMBER OF MODES
    N_r = np.shape(Phi)[1]

    # CUMULATIVE ENERGY AND ENERGY INITAILIZATION
    cum_energy = {'sigma':  np.zeros((N_r)), 'acc':  np.zeros((N_r))}
    energy =  {'sigma':  np.zeros((N_r)), 'acc':  np.zeros((N_r))}

    # GET ENERGY OF FLOW RECONSTRUCTION UP TO ith MODE
    for i in range(N_r):
        Dr = np.dot(Phi[:,0:i+1], a[0:i+1, :])

        cum_energy['sigma'][i], cum_energy['acc'][i] = 1 - np.sum(D ** 2 - Dr ** 2) / np.sum(D ** 2), 1 - np.sum((D - Dr) ** 2) / np.sum(D ** 2)

        if i == 0:
            energy['sigma'][i], energy['acc'][i] = cum_energy['sigma'][i], cum_energy['acc'][i]
        else:
            energy['sigma'][i], energy['acc'][i] = cum_energy['sigma'][i] - cum_energy['sigma'][i-1], cum_energy['acc'][i] - cum_energy['acc'][i-1]

    return cum_energy, energy

def energy_AE(D, out_AE, flag_AE, AE):
    """
    Orders AE modes by energy content

    :param D: snapshot matrix (N_v, N_t)
    :param out_AE: AE modes (if MD-CNN-AE or CNN-HAE) or latent space (C-CNN-AE or CNN-VAE)
    :param flag_AE: type of AE flag
    :param AE: AE model class
    :return: cumulative energy, energy arrays and indices of energy ordering
    """

    # NUMBER OF MODES DEPENDING ON INPUT VARIABLE
    if (flag_AE=='MD-CNN-AE') or (flag_AE=='CNN-HAE'):
        Phi = out_AE
        N_z = np.shape(Phi)[2]
        del out_AE
    elif (flag_AE=='CNN-VAE') or (flag_AE=='C-CNN-AE'):
        z = out_AE
        z_aux = np.zeros(np.shape(z.numpy()))
        N_z = np.shape(z)[1]
        del out_AE

    # INITIALIZATE CUMULATIVE ENERGY, ENERGY AND MODE ORDERING
    cum_energy = {'sigma':  np.zeros((N_z)), 'acc':  np.zeros((N_z))}
    energy =  {'sigma':  np.zeros((N_z)), 'acc':  np.zeros((N_z))}
    i_energy_AE = {'sigma': [], 'acc': []}

    i_unordered_sigma = [*range(N_z)]
    i_unordered_acc = [*range(N_z)]

    # GET ENERGY FOR FLOW RECONSTRUCTION UP TO ith MODE
    for i in range(N_z):

        if flag_AE=='CNN-HAE':

            Dr_sigma = Phi[:, :, i]
            Dr_acc = Phi[:, :, i]

        else:

            err_sigma = np.zeros((N_z-i))
            err_acc = np.zeros((N_z-i))

            count = 0
            for j in i_unordered_sigma:

                if (flag_AE=='CNN-VAE') or (flag_AE=='C-CNN-AE'):

                    z_aux[:, i_energy_AE['sigma'] + [j]] = z.numpy()[:, i_energy_AE['sigma'] + [j]]
                    Dr_sigma = AE.decoder(tf.convert_to_tensor(z_aux)).numpy()
                    Dr_sigma = CNNAE2raw(Dr_sigma)

                    z_aux = np.zeros(np.shape(z.numpy()))

                elif flag_AE=='MD-CNN-AE':

                    Dr_sigma = np.sum(Phi[:,:,i_energy_AE['sigma'] + [j]], axis=2)

                err_sigma[count] = np.sum(D ** 2 - Dr_sigma ** 2) / np.sum(D ** 2)
                count += 1

            count = 0
            for j in i_unordered_acc:

                if (flag_AE == 'CNN-VAE') or (flag_AE == 'C-CNN-AE'):

                    z_aux[:, i_energy_AE['acc'] + [j]] = z.numpy()[:, i_energy_AE['acc'] + [j]]
                    Dr_acc = AE.decoder(tf.convert_to_tensor(z_aux)).numpy()
                    Dr_acc = CNNAE2raw(Dr_acc)

                    z_aux = np.zeros(np.shape(z.numpy()))

                elif flag_AE == 'MD-CNN-AE':

                    Dr_acc = np.sum(Phi[:, :, i_energy_AE['acc'] + [j]], axis=2)

                err_sigma[count] = np.sum(D ** 2 - Dr_acc ** 2) / np.sum(D ** 2)
                count += 1

            i_energy_AE['sigma'].append(i_unordered_sigma[np.argmin(err_sigma)])
            i_energy_AE['acc'].append(i_unordered_acc[np.argmin(err_acc)])

            i_unordered_sigma.remove(i_unordered_sigma[np.argmin(err_sigma)])
            i_unordered_acc.remove(i_unordered_acc[np.argmin(err_acc)])

            if (flag_AE == 'CNN-VAE') or (flag_AE == 'C-CNN-AE'):

                z_aux[:, i_energy_AE['acc']] = z.numpy()[:, i_energy_AE['acc']]
                Dr_acc = AE.decoder(tf.convert_to_tensor(z_aux)).numpy()
                Dr_acc = CNNAE2raw(Dr_acc)
                z_aux = np.zeros(np.shape(z.numpy()))

                z_aux[:, i_energy_AE['sigma']] = z.numpy()[:, i_energy_AE['sigma']]
                Dr_sigma = AE.decoder(tf.convert_to_tensor(z_aux)).numpy()
                Dr_sigma = CNNAE2raw(Dr_sigma)
                z_aux = np.zeros(np.shape(z.numpy()))

            elif flag_AE == 'MD-CNN-AE':

                Dr_sigma = np.sum(Phi[:,:,i_energy_AE['sigma']], axis=2)
                Dr_acc = np.sum(Phi[:, :, i_energy_AE['acc']], axis=2)


        cum_energy['sigma'][i], cum_energy['acc'][i] = 1 - np.sum(D ** 2 - Dr_sigma ** 2) / np.sum(D ** 2), 1 - np.sum((D - Dr_acc) ** 2) / np.sum(D ** 2)
        if i == 0:
            energy['sigma'][i], energy['acc'][i] = cum_energy['sigma'][i], cum_energy['acc'][i]
        else:
            energy['sigma'][i], energy['acc'][i] = cum_energy['sigma'][i] - cum_energy['sigma'][i - 1], cum_energy['acc'][i] - cum_energy['acc'][i - 1]

    return cum_energy, energy, i_energy_AE

