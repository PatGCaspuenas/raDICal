import numpy as np
import sklearn.metrics


def get_RMSE(Dtrue, D, B, flag_type):

    # PARAMETERS
    m = np.shape(B)[0]
    n = np.shape(B)[1]

    # GET DATA OUTSIDE MASK
    B = np.reshape(B, (m*n), order='F')
    if np.shape(D)[0] == m * n:
        i_nonmask = np.where(np.isnan(B))
    elif np.shape(D)[0] == 2 * m * n:
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

    # PARAMETERS
    m = np.shape(B)[0]
    n = np.shape(B)[1]

    # GET DATA OUTSIDE MASK
    B = np.reshape(B, (m*n), order='F')
    if np.shape(D)[0] == m * n:
        i_nonmask = np.where(np.isnan(B))
    elif np.shape(D)[0] == 2 * m * n:
        i_nonmask = np.where(np.isnan(np.concatenate((B, B))))
    else:
        i_nonmask = np.where(np.isnan(np.concatenate((B,B,B))))

    Xtrue = Dtrue[i_nonmask, :]
    X = D[i_nonmask, :]

    # COMPUTE CUMULATIVE ENERGETIC ACCURACY
    CEA = 1 - np.sum((Xtrue - X) ** 2) / np.sum(Xtrue ** 2)

    return CEA

def get_R2factor(Xtrue, X, flag_R2method):

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

def get_cos_similarity(Dtrue, D, B):

    # PARAMETERS
    m = np.shape(B)[0]
    n = np.shape(B)[1]
    nt = np.shape(Dtrue)[1]

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


    B = np.reshape(B, (m*n), order='F')
    if np.shape(D)[0] == m * n:
        i_nonmask = np.where(np.isnan(B))
    elif np.shape(D)[0] == 2 * m * n:
        i_nonmask = np.where(np.isnan(np.concatenate((B, B))))
    else:
        i_nonmask = np.where(np.isnan(np.concatenate((B,B,B))))

    Xtrue = Dtrue[i_nonmask, :]
    X = D[i_nonmask, :]
    Sc = 0
    for t in range(nt):
        Sc = Sc + np.sum( np.multiply(Xtrue[0,:,t],X[0,:,t]) / (np.linalg.norm(Xtrue[0,:,t])) / (np.linalg.norm(X[0,:,t])) )
    Sc =  Sc / nt

    return Sc


def get_latent_correlation_matrix(z_test):

    nr = np.shape(z_test)[1]
    nt = np.shape(z_test)[0]

    Rij = np.zeros((nr, nr))

    Cij = np.abs(np.cov(z_test.T))

    for i in range(nr):
        for j in range(nr):
            Rij[i,j] = Cij[i,j] / np.sqrt(Cij[i,i] * Cij[j,j])

    detR = np.linalg.det(Rij)

    return detR, Rij

def get_latent_MI(z_test):

    nr = np.shape(z_test)[1]
    MI = np.zeros((nr, nr))

    for i in range(nr):
        for j in range(i,nr):

            MI[i,j] = sklearn.metrics.mutual_info_score(z_test[:, i], z_test[:, j])

    return MI



