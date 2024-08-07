# PACKAGES
import numpy as np
import sklearn.metrics

# LOCAL FUNCTIONS
from utils.data.transformer import window2flow

def get_RMSE(Dtrue, D, B, flag_type):
    """
    Estimates Root Mean Square Error (normalized with standard deviation of ground truth flow) for snapshot matrix
    :param Dtrue: ground truth of snapshot matrix (N_v, N_t) or (N_w, N_v, N_t)
    :param D: reconstructed snapshot matrix (N_v, N_t) or (N_w, N_v, N_t)
    :param B: mask grid (1 if body, 0 otherwise)
    :param flag_type: 'W' if whole error, 'S' if spatial, 'T' if temporal
    :return: RMSE
    """

    # If flow is in PW format, convert to raw format
    if Dtrue.ndim == 3:
        Dtrue = window2flow(Dtrue)
        D = window2flow(D)

    # PARAMETERS
    N_y, N_x = np.shape(B)

    # GET DATA OUTSIDE MASK
    B = np.reshape(B, (N_y * N_x), order='F')
    if np.shape(D)[0] == (N_y * N_x):
        i_nonmask = np.where(np.isnan(B))
    elif np.shape(D)[0] == 2 * (N_y * N_x):
        i_nonmask = np.where(np.isnan(np.concatenate((B, B))))
    else:
        i_nonmask = np.where(np.isnan(np.concatenate((B,B,B))))

    Xtrue = Dtrue[i_nonmask, :]
    X = D[i_nonmask, :]

    # STANDARD DEVIATION OF GROUND TRUTH
    std_true = np.std(Xtrue)

    # COMPUTE TEMPORAL (T), SPATIAL (S) OR WHOLE (W) ERROR
    if flag_type == 'T':
        RMSE = np.sqrt(np.mean((Xtrue - X)**2, axis=0)) / std_true
    elif flag_type == 'S':
        RMSE = np.sqrt(np.mean((Xtrue - X) ** 2, axis=1)) / std_true
    elif flag_type == 'W':
        RMSE = np.sqrt(np.mean((Xtrue - X) ** 2)) / std_true

    return RMSE

def get_CEA(Dtrue, D, B):
    """
    Estimates Cumulative Energetic Accuracy (similar to CE) for snapshot matrix
    :param Dtrue: ground truth of snapshot matrix (N_v, N_t) or (N_w, N_v, N_t)
    :param D: reconstructed snapshot matrix (N_v, N_t) or (N_w, N_v, N_t)
    :param B: mask grid (1 if body, 0 otherwise)
    :return: CEA
    """

    # If flow is in PW format, convert to raw format
    if Dtrue.ndim == 3:
        Dtrue = window2flow(Dtrue)
        D = window2flow(D)

    # PARAMETERS
    N_y, N_x = np.shape(B)

    # GET DATA OUTSIDE MASK
    B = np.reshape(B, (N_y * N_x), order='F')
    if np.shape(D)[0] == (N_y * N_x):
        i_nonmask = np.where(np.isnan(B))
    elif np.shape(D)[0] == 2 * (N_y * N_x):
        i_nonmask = np.where(np.isnan(np.concatenate((B, B))))
    else:
        i_nonmask = np.where(np.isnan(np.concatenate((B,B,B))))

    Xtrue = Dtrue[i_nonmask, :]
    X = D[i_nonmask, :]

    # COMPUTE CUMULATIVE ENERGETIC ACCURACY
    CEA = 1 - np.sum((Xtrue - X) ** 2) / np.sum(Xtrue ** 2)

    return CEA

def get_cos_similarity(Dtrue, D, B):
    """
    Estimates temporal cosine similarity Sc for snapshot matrix
    :param Dtrue: ground truth of snapshot matrix (N_v, N_t) or (N_w, N_v, N_t)
    :param D: reconstructed snapshot matrix (N_v, N_t) or (N_w, N_v, N_t)
    :param B: mask grid (1 if body, 0 otherwise)
    :return: Sc
    """

    # If flow is in PW format, convert to raw format
    if Dtrue.ndim == 3:
        Dtrue = window2flow(Dtrue)
        D = window2flow(D)

    # PARAMETERS
    N_y, N_x = np.shape(B)
    N_t = np.shape(Dtrue)[1]

    # GET DATA OUTSIDE MASK
    B = np.reshape(B, (N_y * N_x), order='F')
    if np.shape(D)[0] == (N_y * N_x):
        i_nonmask = np.where(np.isnan(B))
    elif np.shape(D)[0] == 2 * (N_y * N_x):
        i_nonmask = np.where(np.isnan(np.concatenate((B, B))))
    else:
        i_nonmask = np.where(np.isnan(np.concatenate((B,B,B))))

    Xtrue = Dtrue[i_nonmask, :]
    X = D[i_nonmask, :]
    Sc = 0
    for t in range(N_t):
        Sc = Sc + np.sqrt(np.sum(np.multiply(Xtrue[0,:,t],X[0,:,t]))) / (np.linalg.norm(Xtrue[0,:,t]))
    Sc =  Sc / N_t

    return Sc

    # Other implementation of Sc
    # GET DATA OUTSIDE MASK
    # i_nonmask = np.where(np.isnan(B))
    # if np.shape(D)[0] == m * n:
    #     k = 1
    # elif np.shape(D)[0] == 2 * m * n:
    #     k = 2
    # else:
    #     k = 3
    # nv_nonmask = np.array(i_nonmask).size / k
    #
    # # RESHAPE FLOW (control is already reshaped)
    # Xtrue = np.zeros((nt, n, m, k))
    # X = np.zeros((nt, n, m, k))
    # SC = np.zeros((nt, n, m))
    # normtrue, norm = np.zeros((2, nt, n, m))
    # for i in range(k):
    #     Xtrue[:, :, :, i] = np.reshape(Dtrue[( (n * m)*i ):( (n * m)*(i + 1) ), :], (m, n, nt), order='F').T
    #     X[:, :, :, i] = np.reshape(D[((n * m) * i):((n * m) * (i + 1)), :], (m, n, nt), order='F').T
    #
    #     SC = SC + np.multiply(Xtrue[:, :, :, i], X[:, :, :, i])
    #     normtrue = normtrue + Xtrue[:, :, :, i]**2
    #     norm = norm + X[:, :, :, i] ** 2
    #
    # SC = SC / np.multiply(np.sqrt(norm), np.sqrt(normtrue))
    #
    # Sc = 0
    # for t in range(nt):
    #     aux = SC[t, :, :].T
    #     Sc =  Sc + np.sum(aux[i_nonmask])
    # Sc = Sc / (nv_nonmask * nt)




