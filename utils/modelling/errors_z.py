import numpy as np
import numpy.fft as fft
import sklearn.metrics

from utils.data.transform_data import z_window2concat

def get_RMSE_z(Xtrue, X, n_p=0):

    if Xtrue.ndim == 3:
        Xtrue = z_window2concat(Xtrue, n_p)
        X = z_window2concat(X, n_p)

    # PARAMETERS
    nr = np.shape(Xtrue)[1]

    RMSE = np.zeros((nr))
    for i in range(nr):
        RMSE[i] = np.sqrt(np.mean((Xtrue[:, i] - X[:, i]) ** 2, axis=0))

    return RMSE

def get_max_ntpred(Xtrue, X, n_p):

    # PARAMETERS
    nW, nt, nr = np.shape(Xtrue)

    ntpred = np.zeros((nW))
    R2threshold = 0.9

    for w in range(nW):
        for t in range(n_p, nt):
            R2 = get_R2factor(Xtrue[w, :t, :], X[w, :t, :], 'C')
            if any(R2 < R2threshold):
                ntpred[w] = t
                break

    ntpred = int(np.mean(ntpred))

    return ntpred

def get_R2factor(Xtrue, X, flag_R2method, n_p =0):

    if Xtrue.ndim == 3:
        Xtrue = z_window2concat(Xtrue, n_p)
        X = z_window2concat(X, n_p)

    # PARAMETERS
    if Xtrue.ndim == 1:
        nr = 1
        Xtrue = np.copy(Xtrue).reshape(-1,1)
        X = np.copy(X).reshape(-1, 1)
    else:
        nr = np.shape(Xtrue)[1]

    R2 = np.zeros(nr)

    # COMPUTE R2 FOR EACH MODE
    for i in range(nr):

        if flag_R2method == 'D': # DETERMINATION
            R2[i] = 1 - np.sum((Xtrue[:,i] - X[:,i]) ** 2) / np.sum((Xtrue[:,i] - np.mean(Xtrue[:,i])) ** 2)

        elif flag_R2method == 'C': # CORRELATION
            R2[i] = np.mean(np.multiply(Xtrue[:,i], X[:,i])) ** 2 / (np.mean(Xtrue[:,i] ** 2) * np.mean(X[:,i] ** 2))

        if np.isnan(R2[i]): # CORRECT FOR INDETERMINATION
            R2[i] = 1e-4

    return R2


def get_latent_correlation_matrix(z_test):

    nr = np.shape(z_test)[1]
    nt = np.shape(z_test)[0]

    Rij = np.zeros((nr, nr))

    Cij = np.abs(np.cov(z_test.T))

    for i in range(nr):
        for j in range(nr):
            Rij[i,j] = Cij[i,j] / np.sqrt(Cij[i,i] * Cij[j,j])

    meanR = 0
    c = 0
    for i in range(nr):
        for j in range(i+1,nr):
            meanR += Rij[i,j]
            c += 1

    meanR /= c
    detR = np.linalg.det(Rij)

    return detR, meanR, Rij

def get_latent_MI(z_test):

    nr = np.shape(z_test)[1]
    MI = np.zeros((nr, nr))

    for i in range(nr):
        for j in range(i,nr):

            MI[i,j] = sklearn.metrics.mutual_info_score(z_test[:, i], z_test[:, j])

    return MI

def get_frequencies(z_test):

    nr = np.shape(z_test)[1]
    peaks = []

    for i in range(nr):
        spectrum = fft.fft(z_test[:, i])
        freq = fft.fftfreq(len(spectrum), d=0.1)
        threshold = 0.5 * max(abs(spectrum))
        mask = (abs(spectrum) > threshold)

        freqaux = freq[mask]
        freqaux = [f for f in freqaux if f > 0]
        peaks.append(freqaux)

    return peaks


