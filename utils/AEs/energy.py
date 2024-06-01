import numpy as np
import tensorflow as tf

from utils.data.transform_data import *

def energy_POD(D_test, Phi, a):

    # NUMBER OF MODES
    nr = np.shape(Phi)[1]

    # CUMULATIVE ENERGY AND ENERGY INITAILIZATION
    cum_energy = {'sigma':  np.zeros((nr)), 'acc':  np.zeros((nr))}
    energy =  {'sigma':  np.zeros((nr)), 'acc':  np.zeros((nr))}

    # GET ENERGY OF FLOW RECONSTRUCTION UP TO ith MODE
    for i in range(nr):
        Dr = np.dot(Phi[:,0:i+1], a[0:i+1, :])

        cum_energy['sigma'][i], cum_energy['acc'][i] = 1 - np.sum(D_test ** 2 - Dr ** 2) / np.sum(D_test ** 2), 1 - np.sum((D_test - Dr) ** 2) / np.sum(D_test ** 2)

        if i == 0:
            energy['sigma'][i], energy['acc'][i] = cum_energy['sigma'][i], cum_energy['acc'][i]
        else:
            energy['sigma'][i], energy['acc'][i] = cum_energy['sigma'][i] - cum_energy['sigma'][i-1], cum_energy['acc'][i] - cum_energy['acc'][i-1]

    return cum_energy, energy

def energy_AE(D_test, out_AE, flag_AE, AE):

    # NUMBER OF MODES DEPENDING ON INPUT VARIABLE
    if (flag_AE=='MD-CNN-AE') or (flag_AE=='CNN-HAE'):
        Phi_test = out_AE
        nr = np.shape(Phi_test)[2]
        del out_AE
    elif (flag_AE=='CNN-VAE') or (flag_AE=='C-CNN-AE'):
        z_test = out_AE
        z_aux = np.zeros(np.shape(z_test.numpy()))
        nr = np.shape(z_test)[1]
        del out_AE

    # INITIALIZATE CUMULATIVE ENERGY, ENERGY AND MODE ORDERING
    cum_energy = {'sigma':  np.zeros((nr)), 'acc':  np.zeros((nr))}
    energy =  {'sigma':  np.zeros((nr)), 'acc':  np.zeros((nr))}
    i_energy_AE = {'sigma': [], 'acc': []}

    i_unordered_sigma = [*range(nr)]
    i_unordered_acc = [*range(nr)]

    # GET ENERGY FOR FLOW RECONSTRUCTION UP TO ith MODE
    for i in range(nr):

        if flag_AE=='CNN-HAE':

            Dr_sigma = Phi_test[:, :, i]
            Dr_acc = Phi_test[:, :, i]

        else:

            err_sigma = np.zeros((nr-i))
            err_acc = np.zeros((nr-i))

            count = 0
            for j in i_unordered_sigma:

                if (flag_AE=='CNN-VAE') or (flag_AE=='C-CNN-AE'):

                    z_aux[:, i_energy_AE['sigma'] + [j]] = z_test.numpy()[:, i_energy_AE['sigma'] + [j]]
                    Dr_sigma = AE.decoder(tf.convert_to_tensor(z_aux)).numpy()
                    Dr_sigma = CNNAE2raw(Dr_sigma)

                    z_aux = np.zeros(np.shape(z_test.numpy()))

                elif flag_AE=='MD-CNN-AE':

                    Dr_sigma = np.sum(Phi_test[:,:,i_energy_AE['sigma'] + [j]], axis=2)

                err_sigma[count] = np.sum(D_test ** 2 - Dr_sigma ** 2) / np.sum(D_test ** 2)
                count += 1

            count = 0
            for j in i_unordered_acc:

                if (flag_AE == 'CNN-VAE') or (flag_AE == 'C-CNN-AE'):

                    z_aux[:, i_energy_AE['acc'] + [j]] = z_test.numpy()[:, i_energy_AE['acc'] + [j]]
                    Dr_acc = AE.decoder(tf.convert_to_tensor(z_aux)).numpy()
                    Dr_acc = CNNAE2raw(Dr_acc)

                    z_aux = np.zeros(np.shape(z_test.numpy()))

                elif flag_AE == 'MD-CNN-AE':

                    Dr_acc = np.sum(Phi_test[:, :, i_energy_AE['acc'] + [j]], axis=2)

                err_sigma[count] = np.sum(D_test ** 2 - Dr_acc ** 2) / np.sum(D_test ** 2)
                count += 1

            i_energy_AE['sigma'].append(i_unordered_sigma[np.argmin(err_sigma)])
            i_energy_AE['acc'].append(i_unordered_acc[np.argmin(err_acc)])

            i_unordered_sigma.remove(i_unordered_sigma[np.argmin(err_sigma)])
            i_unordered_acc.remove(i_unordered_acc[np.argmin(err_acc)])

            if (flag_AE == 'CNN-VAE') or (flag_AE == 'C-CNN-AE'):

                z_aux[:, i_energy_AE['acc']] = z_test.numpy()[:, i_energy_AE['acc']]
                Dr_acc = AE.decoder(tf.convert_to_tensor(z_aux)).numpy()
                Dr_acc = CNNAE2raw(Dr_acc)
                z_aux = np.zeros(np.shape(z_test.numpy()))

                z_aux[:, i_energy_AE['sigma']] = z_test.numpy()[:, i_energy_AE['sigma']]
                Dr_sigma = AE.decoder(tf.convert_to_tensor(z_aux)).numpy()
                Dr_sigma = CNNAE2raw(Dr_sigma)
                z_aux = np.zeros(np.shape(z_test.numpy()))

            elif flag_AE == 'MD-CNN-AE':

                Dr_sigma = np.sum(Phi_test[:,:,i_energy_AE['sigma']], axis=2)
                Dr_acc = np.sum(Phi_test[:, :, i_energy_AE['acc']], axis=2)


        cum_energy['sigma'][i], cum_energy['acc'][i] = 1 - np.sum(D_test ** 2 - Dr_sigma ** 2) / np.sum(D_test ** 2), 1 - np.sum((D_test - Dr_acc) ** 2) / np.sum(D_test ** 2)
        if i == 0:
            energy['sigma'][i], energy['acc'][i] = cum_energy['sigma'][i], cum_energy['acc'][i]
        else:
            energy['sigma'][i], energy['acc'][i] = cum_energy['sigma'][i] - cum_energy['sigma'][i - 1], cum_energy['acc'][i] - cum_energy['acc'][i - 1]

    return cum_energy, energy, i_energy_AE

