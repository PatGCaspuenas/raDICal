import numpy as np

def get_RMSE(Dtrue, D, B, flag_type):

    m = np.shape(B)[0]
    n = np.shape(B)[1]
    B = np.reshape(B, (m*n), order='F')
    i_nonmask = np.where(np.isnan(B))

    Xtrue = Dtrue[i_nonmask, :]
    X = D[i_nonmask, :]

    std_true = np.std(Xtrue)

    if flag_type == 'T':
        RMSE = np.sqrt(np.mean((Xtrue - X)**2, axis=0)) / std_true
    elif flag_type == 'S':
        RMSE = np.sqrt(np.mean((Xtrue - X) ** 2, axis=1)) / std_true
    elif flag_type == 'W':
        RMSE = np.sqrt(np.mean((Xtrue - X) ** 2)) / std_true

    return RMSE

def get_R2factor(Xtrue, X, flag_R2method):

    if len(np.shape(Xtrue)) == 1:
        nr = 1
    else:
        nr = np.shape(Xtrue)[1]

    R2 = np.zeros(nr)

    for i in range(nr):

        if flag_R2method == 'D':
            if nr == 1:
                R2[i] = 1 - np.sum((Xtrue - X)**2) / np.sum((Xtrue - np.mean(Xtrue))**2)
            else:
                R2[i] = 1 - np.sum((Xtrue[:,i] - X[:,i]) ** 2) / np.sum((Xtrue[:,i] - np.mean(Xtrue[:,i])) ** 2)

        elif flag_R2method == 'C':
            if nr == 1:
                R2[i] = np.mean(np.multiply(Xtrue, X))**2 / (np.mean(Xtrue**2)*np.mean(X**2))
            else:
                R2[i] = np.mean(np.multiply(Xtrue[:,i], X[:,i])) ** 2 / (np.mean(Xtrue[:,i] ** 2) * np.mean(X[:,i] ** 2))

        if np.isnan(R2[i]):
            R2[i] = 1e-4

    return R2

