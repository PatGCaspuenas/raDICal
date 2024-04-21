import numpy as np

def diff_time(D,t):

    dt = t[1] - t[0]
    nt = np.shape(D)[1]
    dDdt = np.zeros(np.shape(D))

    # Central difference
    dDdt[:, 1:-1] = (D[:, 2:] - D[:, 0:-2]) / (2 * dt)

    # Forward difference
    dDdt[:, 0] = ( - 3*D[:, 0] + 4*D[:, 1] - D[:, 2] ) / (2 * dt)

    # Backward difference
    dDdt[:, -1] = -(- 3 * D[:, -1] + 4 * D[:, -2] - D[:, -3]) / (2 * dt)

    return dDdt

def diff_1st_2Dspace(X,Y,D):

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(D)[1]

    dx = np.abs(X[0,1] - X[0,0])
    dy = np.abs(Y[0,0] - Y[1,0])

    U = np.reshape(D[0:m * n,:], (m, n, nt), order='F')
    V = np.reshape(D[m * n:2 * m* n, :], (m, n, nt), order='F')

    Ux, Uy, Vx, Vy = np.zeros((4, m, n, nt))

    Ux[:, 1:-1, :] = (U[:, 2:, :] - U[:, 0:-2, :]) / (2 * dx)
    Ux[:, 0, :] = (-3 * U[:, 0, :] + 4 * U[:, 1, :] - U[:, 2, :]) / (2 * dx)
    Ux[:, -1, :] = (3 * U[:, -1, :] - 4 * U[:, -2, :] + U[:, -3, :]) / (2 * dx)

    Uy[1:-1, :, :] = (U[2:, :, :] - U[0:-2, :, :]) / (2 * dy)
    Uy[0, :, :] = (-3 * U[0, :, :] + 4 * U[1, :, :] - U[2, :, :]) / (2 * dy)
    Uy[-1, :, :] = (3 * U[-1, :, :] - 4 * U[-2, :, :] + U[-3, :, :]) / (2 * dy)

    Vx[:, 1:-1, :] = (V[:, 2:, :] - V[:, 0:-2, :]) / (2 * dx)
    Vx[:, 0, :] = (-3 * V[:, 0, :] + 4 * V[:, 1, :] - V[:, 2, :]) / (2 * dx)
    Vx[:, -1, :] = (3 * V[:, -1, :] - 4 * V[:, -2, :] + V[:, -3, :]) / (2 * dx)

    Vy[1:-1, :, :] = (V[2:, :, :] - V[0:-2, :, :]) / (2 * dy)
    Vy[0, :, :] = (-3 * V[0, :, :] + 4 * V[1, :, :] - V[2, :, :]) / (2 * dy)
    Vy[-1, :, :] = (3 * V[-1, :, :] - 4 * V[-2, :, :] + V[-3, :, :]) / (2 * dy)

    Dx = np.concatenate((np.reshape(Ux, (m * n, nt), order='F'), np.reshape(Vx, (m * n, nt), order='F')), axis=0)
    Dy = np.concatenate((np.reshape(Uy, (m * n, nt), order='F'), np.reshape(Vy, (m * n, nt), order='F')), axis=0)

    return Dx, Dy

def get_2Dvorticity(X,Y,D):

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    Dx, Dy = diff_1st_2Dspace(X,Y,D)

    Vx = Dx[m * n:2 * m * n, :]
    Uy = Dy[0:m * n, :]

    w = Vx - Uy

    return w

