# PACKAGES
import numpy as np
import numpy.fft as fft
import sklearn.metrics

# LOCAL FUNCTIONS
from utils.data.transformer import window2zcat

def get_RMSE_z(Xtrue, X, w_prop=0):
    """
    Estimation of Root Mean Square Error (normalized with standard deviation of ground truth latent coordinate)
    for latent space up to a certain length of the PW

    :param Xtrue: ground truth of latent space (N_t, N_z) or (N_w, N_t, N_z)
    :param X: reconstructed latent space (N_t, N_z) or (N_w, N_t, N_z)
    :param w_prop: PW window to evaluate RMSE
    :return: RMSE for each latent coordinate
    """

    # If latent space is given in PW shape, transform to raw shape
    if Xtrue.ndim == 3:
        Xtrue = window2zcat(Xtrue, w_prop)
        X = window2zcat(X, w_prop)

    # PARAMETERS
    N_z = np.shape(Xtrue)[1]

    # Evaluate RMSE for each latent coordinate
    RMSE = np.zeros((N_z))
    for i in range(N_z):
        std_i = np.std(Xtrue[:, i])
        RMSE[i] = np.sqrt(np.mean((Xtrue[:, i] - X[:, i]) ** 2, axis=0)) / std_i

    return RMSE

def get_max_w_prop(Xtrue, X, w_p):
    """
    Estimation of number of time instants where prediction is above a certain R2 threshold

    :param Xtrue: ground truth of latent space (N_w, N_t, N_z)
    :param X: reconstructed latent space (N_w, N_t, N_z)
    :param w_p: PW window to evaluate RMSE
    :return: number of prediction time instants where condition holds
    """

    # PARAMETERS
    N_w, N_t, N_z = np.shape(Xtrue)

    w_prop_s = np.zeros((N_w))
    R2threshold = 0.9

    # For each window, number of time instants is evaluated. Mean R2 condition is applied
    for w in range(N_w):
        for t in range(w_p, N_t):
            R2 = get_R2factor(Xtrue[w, :t, :], X[w, :t, :], 'C')
            if (np.mean(R2) < R2threshold):
                w_prop_s[w] = t
                break

    # Mean over all windows
    w_prop_s = int(np.mean(w_prop_s))

    return w_prop_s

def get_R2factor(Xtrue, X, flag_R2method, w_prop =0):
    """
    Estimates R2 factor for latent space up to a certain length of the PW

    :param Xtrue: ground truth of latent space (N_t, N_z) or (N_w, N_t, N_z)
    :param X: reconstructed latent space (N_t, N_z) or (N_w, N_t, N_z)
    :param w_prop: PW window to evaluate RMSE
    :param flag_R2method: 'D' if deterministic, 'C' if correlation
    :return: R2 factor for each latent coordinate
    """

    # If latent space is given in PW shape, transform to raw shape
    if Xtrue.ndim == 3:
        Xtrue = window2zcat(Xtrue, w_prop)
        X = window2zcat(X, w_prop)

    # PARAMETERS
    if Xtrue.ndim == 1:
        N_z = 1
        Xtrue = np.copy(Xtrue).reshape(-1,1)
        X = np.copy(X).reshape(-1, 1)
    else:
        N_z = np.shape(Xtrue)[1]

    R2 = np.zeros(N_z)

    # COMPUTE R2 FOR EACH MODE
    for i in range(N_z):

        if flag_R2method == 'D': # DETERMINATION
            R2[i] = 1 - np.sum((Xtrue[:,i] - X[:,i]) ** 2) / np.sum((Xtrue[:,i] - np.mean(Xtrue[:,i])) ** 2)

        elif flag_R2method == 'C': # CORRELATION
            R2[i] = np.mean(np.multiply(Xtrue[:,i], X[:,i])) ** 2 / (np.mean(Xtrue[:,i] ** 2) * np.mean(X[:,i] ** 2))

        if np.isnan(R2[i]): # CORRECT FOR INDETERMINATION
            R2[i] = 1e-4

    return R2


def get_latent_correlation_matrix(z):
    """
    Obtains correlation matrix for latent space

    :param z: latent space (N_t, N_z)
    :return: determinant of correlation matrix, mean of corr matrix and correlation matrx
    """

    # Parameters
    N_z = np.shape(z)[1]
    N_t = np.shape(z)[0]

    Rij = np.zeros((N_z, N_z))

    # Correlation matrix
    Cij = np.abs(np.cov(z.T))

    for i in range(N_z):
        for j in range(N_z):
            Rij[i,j] = Cij[i,j] / np.sqrt(Cij[i,i] * Cij[j,j])

    # Mean of correlation matrix for upper triangle of the latter
    meanR = 0
    c = 0
    for i in range(N_z):
        for j in range(i+1,N_z):
            meanR += Rij[i,j]
            c += 1

    meanR /= c
    detR = np.linalg.det(Rij)

    return detR, meanR, Rij

def get_latent_MI(z):
    """
    Estimation of Mutual Information (nonlinear correlation) for latent space

    :param z: latent space (N_t, N_z)
    :return: Mutual information matrix
    """

    # Parameters
    N_z = np.shape(z)[1]
    MI = np.zeros((N_z, N_z))

    # Mutual Information matrix
    for i in range(N_z):
        for j in range(i,N_z):

            MI[i,j] = sklearn.metrics.mutual_info_score(z[:, i], z[:, j])

    return MI

def get_frequencies(z):
    """
    Estimation of most relevant frequencies for each latent coordinate

    :param z: latent space (N_t, N_z)
    :return: frequency array
    """

    # Parameters
    N_z = np.shape(z)[1]
    peaks = []

    # FFT of each coordinate and frequency retrieval above a certain threhold (50% of highest peak)
    for i in range(N_z):
        spectrum = fft.fft(z[:, i])
        freq = fft.fftfreq(len(spectrum), d=0.1)
        threshold = 0.5 * max(abs(spectrum))
        mask = (abs(spectrum) > threshold)

        freqaux = freq[mask]
        freqaux = [f for f in freqaux if f > 0]
        peaks.append(freqaux)

    return peaks


