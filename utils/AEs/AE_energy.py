import numpy as np
import tensorflow as tf

from utils.data.transform_data import *

def energy_MDCNNAE(D_test, Phi_AE, Phi_POD, a_POD):

    nr = np.shape(Phi_AE)[2]

    cum_energy = {'sigma': {'AE': np.zeros((nr)), 'POD': np.zeros((nr))}, 'acc': {'AE': np.zeros((nr)), 'POD': np.zeros((nr))}}
    energy = {'sigma': {'AE': np.zeros((nr)), 'POD': np.zeros((nr))}, 'acc': {'AE': np.zeros((nr)), 'POD': np.zeros((nr))}}
    i_energy_AE = {'sigma': [], 'acc': []}

    # Define POD energy and cumulative energy
    for i in range(nr):
        Dr = np.dot(Phi_POD[:,0:i+1], a_POD[0:i+1, :])
        cum_energy['sigma']['POD'][i], cum_energy['acc']['POD'][i] = 1 - np.sum(D_test ** 2 - Dr ** 2) / np.sum(D_test ** 2), 1 - np.sum((D_test - Dr) ** 2) / np.sum(D_test ** 2)
        if i == 0:
            energy['sigma']['POD'][i], energy['acc']['POD'][i] = cum_energy['sigma']['POD'][i], cum_energy['acc']['POD'][i]
        else:
            energy['sigma']['POD'][i], energy['acc']['POD'][i] = cum_energy['sigma']['POD'][i] - cum_energy['sigma']['POD'][i-1], \
            cum_energy['acc']['POD'][i] - cum_energy['acc']['POD'][i-1]

    # Order and get energy and cum energy of AE modes
    i_unordered_sigma = [*range(nr)]
    i_unordered_acc = [*range(nr)]
    for i in range(nr):

        err_sigma = np.zeros((nr-i))
        err_acc = np.zeros((nr-i))

        count = 0
        for j in i_unordered_sigma:
            err_sigma[count] = np.sum(D_test ** 2 - np.sum(Phi_AE[:,:,i_energy_AE['sigma'] + [j]], axis=2) ** 2) / np.sum(D_test ** 2)
            count = count + 1
        count = 0
        for j in i_unordered_acc:
            err_acc[count] = np.sum((D_test - np.sum(Phi_AE[:,:,i_energy_AE['acc'] + [j]], axis=2)) ** 2) / np.sum(D_test ** 2)
            count = count + 1

        i_energy_AE['sigma'].append(i_unordered_sigma[np.argmin(err_sigma)])
        i_energy_AE['acc'].append(i_unordered_acc[np.argmin(err_acc)])

        i_unordered_sigma.remove(i_unordered_sigma[np.argmin(err_sigma)])
        i_unordered_acc.remove(i_unordered_acc[np.argmin(err_acc)])

        Dr_sigma = np.sum(Phi_AE[:,:,i_energy_AE['sigma']], axis=2)
        Dr_acc = np.sum(Phi_AE[:, :, i_energy_AE['acc']], axis=2)
        cum_energy['sigma']['AE'][i], cum_energy['acc']['AE'][i] = 1 - np.sum(D_test ** 2 - Dr_sigma ** 2) / np.sum(
            D_test ** 2), 1 - np.sum((D_test - Dr_acc) ** 2) / np.sum(D_test ** 2)
        if i == 0:
            energy['sigma']['AE'][i], energy['acc']['AE'][i] = cum_energy['sigma']['AE'][i], \
            cum_energy['acc']['AE'][i]
        else:
            energy['sigma']['AE'][i], energy['acc']['AE'][i] = cum_energy['sigma']['AE'][i] - \
                                                                 cum_energy['sigma']['AE'][i - 1], \
                                                                 cum_energy['acc']['AE'][i] - cum_energy['acc']['AE'][
                                                                     i - 1]
    return cum_energy, energy, i_energy_AE

def energy_CNNHAE(D_test, Phi_AE, Phi_POD, a_POD):

    nr = np.shape(Phi_AE)[2]

    cum_energy = {'sigma': {'AE': np.zeros((nr)), 'POD': np.zeros((nr))}, 'acc': {'AE': np.zeros((nr)), 'POD': np.zeros((nr))}}
    energy = {'sigma': {'AE': np.zeros((nr)), 'POD': np.zeros((nr))}, 'acc': {'AE': np.zeros((nr)), 'POD': np.zeros((nr))}}

    # Define POD energy and cumulative energy
    for i in range(nr):
        Dr = np.dot(Phi_POD[:,0:i+1], a_POD[0:i+1, :])
        cum_energy['sigma']['POD'][i], cum_energy['acc']['POD'][i] = 1 - np.sum(D_test ** 2 - Dr ** 2) / np.sum(D_test ** 2), 1 - np.sum((D_test - Dr) ** 2) / np.sum(D_test ** 2)

        Dr_sigma = np.sum(Phi_AE[:, :, 0:i+1], axis=2)
        Dr_acc = np.sum(Phi_AE[:, :, 0:i+1], axis=2)
        cum_energy['sigma']['AE'][i], cum_energy['acc']['AE'][i] = 1 - np.sum(D_test ** 2 - Dr_sigma ** 2) / np.sum(
            D_test ** 2), 1 - np.sum((D_test - Dr_acc) ** 2) / np.sum(D_test ** 2)
        if i == 0:
            energy['sigma']['POD'][i], energy['acc']['POD'][i] = cum_energy['sigma']['POD'][i], cum_energy['acc']['POD'][i]

            energy['sigma']['AE'][i], energy['acc']['AE'][i] = cum_energy['sigma']['AE'][i], cum_energy['acc']['AE'][i]
        else:
            energy['sigma']['POD'][i], energy['acc']['POD'][i] = cum_energy['sigma']['POD'][i] - cum_energy['sigma']['POD'][i-1], \
            cum_energy['acc']['POD'][i] - cum_energy['acc']['POD'][i-1]

            energy['sigma']['AE'][i], energy['acc']['AE'][i] = cum_energy['sigma']['AE'][i] - cum_energy['sigma']['AE'][i - 1], \
                                                               cum_energy['acc']['AE'][i] - cum_energy['acc']['AE'][i - 1]

    return cum_energy, energy

def energy_CNNVAE(VAE, D_test, z_test, Phi_POD, a_POD):

    nr = np.shape(z_test)[1]
    z_aux = np.zeros(np.shape(z_test.numpy()))

    cum_energy = {'sigma': {'AE': np.zeros((nr)), 'POD': np.zeros((nr))}, 'acc': {'AE': np.zeros((nr)), 'POD': np.zeros((nr))}}
    energy = {'sigma': {'AE': np.zeros((nr)), 'POD': np.zeros((nr))}, 'acc': {'AE': np.zeros((nr)), 'POD': np.zeros((nr))}}
    i_energy_AE = {'sigma': [], 'acc': []}

    # Define POD energy and cumulative energy
    for i in range(nr):
        Dr = np.dot(Phi_POD[:,0:i+1], a_POD[0:i+1, :])
        cum_energy['sigma']['POD'][i], cum_energy['acc']['POD'][i] = 1 - np.sum(D_test ** 2 - Dr ** 2) / np.sum(D_test ** 2), 1 - np.mean(np.sum((D_test - Dr) ** 2, axis = 0)) / np.mean(np.sum(D_test ** 2, axis=0))
        if i == 0:
            energy['sigma']['POD'][i], energy['acc']['POD'][i] = cum_energy['sigma']['POD'][i], cum_energy['acc']['POD'][i]
        else:
            energy['sigma']['POD'][i], energy['acc']['POD'][i] = cum_energy['sigma']['POD'][i] - cum_energy['sigma']['POD'][i-1], \
            cum_energy['acc']['POD'][i] - cum_energy['acc']['POD'][i-1]

    # Order and get energy and cum energy of AE modes
    i_unordered_sigma = [*range(nr)]
    i_unordered_acc = [*range(nr)]
    for i in range(nr):
        print('i = ' + str(i))
        err_sigma = np.zeros((nr-i))
        err_acc = np.zeros((nr-i))

        count = 0
        for j in i_unordered_sigma:
            z_aux[:, i_energy_AE['sigma'] + [j]] = z_test.numpy()[:, i_energy_AE['sigma'] + [j]]
            Dr_sigma = VAE.decoder(tf.convert_to_tensor(z_aux)).numpy()
            Dr_sigma = CNNAE2raw(Dr_sigma)
            err_sigma[count] = np.sum(D_test ** 2 - Dr_sigma ** 2) / np.sum(D_test ** 2)

            z_aux = np.zeros(np.shape(z_test.numpy()))
            count = count + 1
        count = 0
        for j in i_unordered_acc:
            z_aux[:, i_energy_AE['acc'] + [j]] = z_test.numpy()[:, i_energy_AE['acc'] + [j]]
            Dr_acc = VAE.decoder(tf.convert_to_tensor(z_aux)).numpy()
            Dr_acc = CNNAE2raw(Dr_acc)
            err_acc[count] = np.sum((D_test - Dr_acc) ** 2) / np.sum(D_test ** 2)
            print('j = ' + str(j) + ':' + str(err_acc[count]))

            z_aux = np.zeros(np.shape(z_test.numpy()))
            count = count + 1

        i_energy_AE['sigma'].append(i_unordered_sigma[np.argmin(err_sigma)])
        i_energy_AE['acc'].append(i_unordered_acc[np.argmin(err_acc)])

        i_unordered_sigma.remove(i_unordered_sigma[np.argmin(err_sigma)])
        i_unordered_acc.remove(i_unordered_acc[np.argmin(err_acc)])

        z_aux[:, i_energy_AE['acc']] = z_test.numpy()[:, i_energy_AE['acc']]
        Dr_acc = VAE.decoder(tf.convert_to_tensor(z_aux)).numpy()
        Dr_acc = CNNAE2raw(Dr_acc)
        z_aux = np.zeros(np.shape(z_test.numpy()))

        z_aux[:, i_energy_AE['sigma']] = z_test.numpy()[:, i_energy_AE['sigma']]
        Dr_sigma = VAE.decoder(tf.convert_to_tensor(z_aux)).numpy()
        Dr_sigma = CNNAE2raw(Dr_sigma)
        z_aux = np.zeros(np.shape(z_test.numpy()))

        cum_energy['sigma']['AE'][i], cum_energy['acc']['AE'][i] = 1 - np.sum(D_test ** 2 - Dr_sigma ** 2) / np.sum(
            D_test ** 2), 1 - np.sum((D_test - Dr_acc) ** 2) / np.sum(D_test ** 2)
        if i == 0:
            energy['sigma']['AE'][i], energy['acc']['AE'][i] = cum_energy['sigma']['AE'][i], cum_energy['acc']['AE'][i]
        else:
            energy['sigma']['AE'][i], energy['acc']['AE'][i] = cum_energy['sigma']['AE'][i] - cum_energy['sigma']['AE'][i - 1], \
                                                                 cum_energy['acc']['AE'][i] - cum_energy['acc']['AE'][i - 1]
        print('energy : ' + str(energy['acc']['AE'][i]) + ', cum energy : ' + str(cum_energy['acc']['AE'][i]))
    return cum_energy, energy, i_energy_AE