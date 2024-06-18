import numpy as np
from utils.data.transform_data import get_mask_boundaries

def diff_time(D,t):

    # PARAMETERS AND INITIALIZATION
    dt = t[1] - t[0]
    nt = np.shape(D)[1]
    dDdt = np.zeros(np.shape(D))

    # CENTRAL DIFFERENCE
    dDdt[:, 1:-1] = (D[:, 2:] - D[:, 0:-2]) / (2 * dt)

    # FORWARD DIFFERENCE
    dDdt[:, 0] = ( - 3*D[:, 0] + 4*D[:, 1] - D[:, 2] ) / (2 * dt)

    # BACKWARD DIFFERENCE
    dDdt[:, -1] = -(- 3 * D[:, -1] + 4 * D[:, -2] - D[:, -3]) / (2 * dt)

    return dDdt

def diff_1st_2D(grid,D):

    # PARAMETERS AND INITIALIZATION
    X = grid['X']
    Y = grid['Y']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(D)[1]

    dx = np.abs(X[0, 1] - X[0, 0])
    dy = np.abs(Y[0, 0] - Y[1, 0])

    U = np.reshape(D[0:m * n,:], (m, n, nt), order='F')
    V = np.reshape(D[m * n:2 * m* n, :], (m, n, nt), order='F')

    Ux, Uy, Vx, Vy = np.zeros((4, m, n, nt))

    # GRADIENTS
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

    # CORRECT FOR POINTS NEAR MASK
    # iR, iT, iL, iB = get_mask_boundaries(grid['B'])
    #
    # Ux[iR[0], iR[1], :] = (-3 * U[iR[0], iR[1], :] + 4 * U[iR[0], iR[1] + 1, :] - U[iR[0], iR[1] + 2, :]) / (2 * dx)
    # Ux[iL[0], iL[1], :] = (3 * U[iL[0], iL[1], :] - 4 * U[iL[0], iL[1] - 1, :] + U[iL[0], iL[1] - 2, :]) / (2 * dx)
    # Vx[iR[0], iR[1], :] = (-3 * V[iR[0], iR[1], :] + 4 * V[iR[0], iR[1] + 1, :] - V[iR[0], iR[1] + 2, :]) / (2 * dx)
    # Vx[iL[0], iL[1], :] = (3 * V[iL[0], iL[1], :] - 4 * V[iL[0], iL[1] - 1, :] + V[iL[0], iL[1] - 2, :]) / (2 * dx)
    #
    # Uy[iT[0], iT[1], :] = (-3 * U[iT[0], iT[1], :] + 4 * U[iT[0] + 1, iT[1], :] - U[iT[0] + 2, iT[1], :]) / (2 * dy)
    # Uy[iB[0], iB[1], :] = (3 * U[iB[0], iB[1], :] - 4 * U[iB[0] - 1, iB[1], :] + U[iB[0] - 2, iB[1], :]) / (2 * dy)
    # Vy[iT[0], iT[1], :] = (-3 * V[iT[0], iT[1], :] + 4 * V[iT[0] + 1, iT[1], :] - V[iT[0] + 2, iT[1], :]) / (2 * dy)
    # Vy[iB[0], iB[1], :] = (3 * V[iB[0], iB[1], :] - 4 * V[iB[0] - 1, iB[1], :] + V[iB[0] - 2, iB[1], :]) / (2 * dy)

    imask = np.where(grid['B'] == 1)
    Ux[imask[0], imask[1], :] = 0
    Uy[imask[0], imask[1], :] = 0
    Vx[imask[0], imask[1], :] = 0
    Vy[imask[0], imask[1], :] = 0

    # RESHAPE
    Dx = np.concatenate((np.reshape(Ux, (m * n, nt), order='F'), np.reshape(Vx, (m * n, nt), order='F')), axis=0)
    Dy = np.concatenate((np.reshape(Uy, (m * n, nt), order='F'), np.reshape(Vy, (m * n, nt), order='F')), axis=0)

    return Dx, Dy

def get_2Dvorticity(grid,D):

    # PARAMETERS
    X = grid['X']
    Y = grid['Y']

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    # GRADIENTS
    Dx, Dy = diff_1st_2D(grid, D)

    Vx = Dx[m * n:2 * m * n, :]
    Uy = Dy[0:m * n, :]

    # PLANAR VORTICITY
    w = Vx - Uy

    return w
def get_laplacian_2D(grid,D):

    # PARAMETERS
    X = grid['X']
    Y = grid['Y']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(D)[1]

    dx = np.abs(X[0, 0] - X[0, 1])
    dy = np.abs(Y[0, 0] - Y[1, 0])

    # INITIALIZATION
    Uxx = np.zeros((m, n, nt))
    Uyy = np.zeros((m, n, nt))
    Vxx = np.zeros((m, n, nt))
    Vyy = np.zeros((m, n, nt))

    U = np.reshape(D[0:n * m, :], (m, n, nt), order='F')
    V = np.reshape(D[n * m:2 * n * m, :], (m, n, nt), order='F')

    # GRADIENTS
    Uxx[:, 1:-1, :] = (U[:, 2:, :] - 2 * U[:, 1:-1, :] + U[:, 0:-2, :]) / dx ** 2
    Uxx[:, 0, :] = (2 * U[:, 0, :] - 5 * U[:, 1, :] + 4 * U[:, 2, :] - U[:, 3, :]) / dx ** 3
    Uxx[:, -1, :] = (2 * U[:, -1, :] - 5 * U[:, -2, :] + 4 * U[:, -3, :] - U[:, -4, :]) / dx ** 3

    Uyy[1:-1, :, :] = (U[2:, :, :] - 2 * U[1:-1, :, :] + U[0:-2, :, :]) / dy ** 2
    Uyy[0, :, :] = (2 * U[0, :, :] - 5 * U[1, :, :] + 4 * U[2, :, :] - U[3, :, :]) / dy ** 3
    Uyy[-1, :, :] = (2 * U[-1, :, :] - 5 * U[-2, :, :] + 4 * U[-3, :, :] - U[-4, :, :]) / dy ** 3

    Vxx[:, 1:-1, :] = (V[:, 2:, :] - 2 * V[:, 1:-1, :] + V[:, 0:-2, :]) / dx ** 2
    Vxx[:, 0, :] = (2 * V[:, 0, :] - 5 * V[:, 1, :] + 4 * V[:, 2, :] - V[:, 3, :]) / dx ** 3
    Vxx[:, -1, :] = (2 * V[:, -1, :] - 5 * V[:, -2, :] + 4 * V[:, -3, :] - V[:, -4, :]) / dx ** 3

    Vyy[1:-1, :, :] = (V[2:, :, :] - 2 * V[1:-1, :, :] + V[0:-2, :, :]) / dy ** 2
    Vyy[0, :, :] = (2 * V[0, :, :] - 5 * V[1, :, :] + 4 * V[2, :, :] - V[3, :, :]) / dy ** 3
    Vyy[-1, :, :] = (2 * V[-1, :, :] - 5 * V[-2, :, :] + 4 * V[-3, :, :] - V[-4, :, :]) / dy ** 3

    # CORRECT FOR POINTS NEAR MASK
    # iR, iT, iL, iB = get_mask_boundaries(grid['B'])
    #
    # Uxx[iR[0], iR[1], :] = (2 * U[iR[0], iR[1], :] - 5 * U[iR[0], iR[1] + 1, :] + 4 * U[iR[0], iR[1] + 2, :] - U[iR[0], iR[1] + 3, :]) / dx ** 3
    # Uxx[iL[0], iL[1], :] = (2 * U[iL[0], iL[1], :] - 5 * U[iL[0], iL[1] - 1, :] + 4 * U[iL[0], iL[1] - 2, :] - U[iL[0], iL[1] - 3, :]) / dx ** 3
    # Vxx[iR[0], iR[1], :] = (2 * V[iR[0], iR[1], :] - 5 * V[iR[0], iR[1] + 1, :] + 4 * V[iR[0], iR[1] + 2, :] - V[iR[0], iR[1] + 3, :]) / dx ** 3
    # Vxx[iL[0], iL[1], :] = (2 * V[iL[0], iL[1], :] - 5 * V[iL[0], iL[1] - 1, :] + 4 * V[iL[0], iL[1] - 2, :] - V[iL[0], iL[1] - 3, :]) / dx ** 3
    #
    # Uyy[iT[0], iT[1], :] = (2 * U[iT[0], iT[1], :] - 5 * U[iT[0] + 1, iT[1], :] + 4 * U[iT[0] + 2, iT[1], :] - U[iT[0] + 3, iT[1], :]) / dy ** 3
    # Uyy[iB[0], iB[1], :] = (2 * U[iB[0], iB[1], :] - 5 * U[iB[0] - 1, iB[1], :] + 4 * U[iB[0] - 2, iB[1], :] - U[iB[0] - 3, iB[1], :]) / dy ** 3
    # Vyy[iT[0], iT[1], :] = (2 * V[iT[0], iT[1], :] - 5 * V[iT[0] + 1, iT[1], :] + 4 * V[iT[0] + 2, iT[1], :] - V[iT[0] + 3, iT[1], :]) / dy ** 3
    # Vyy[iB[0], iB[1], :] = (2 * V[iB[0], iB[1], :] - 5 * V[iB[0] - 1, iB[1], :] + 4 * V[iB[0] - 2, iB[1], :] - V[iB[0] - 3, iB[1], :]) / dy ** 3

    imask = np.where(grid['B'] == 1)
    Uxx[imask[0], imask[1], :] = 0
    Uyy[imask[0], imask[1], :] = 0
    Vxx[imask[0], imask[1], :] = 0
    Vyy[imask[0], imask[1], :] = 0

    # LAPLACIAN
    Dxx = np.concatenate((np.reshape(Uxx,(n*m,nt),order='F'),np.reshape(Vxx,(n*m,nt),order='F')),axis=0)
    Dyy = np.concatenate((np.reshape(Uyy,(n*m,nt),order='F'),np.reshape(Vyy,(n*m,nt),order='F')),axis=0)

    return Dxx + Dyy

def get_divergence_2D(grid,D):

    # PARAMETERS
    X = grid['X']
    Y = grid['Y']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(D)[1]

    dx = np.abs(X[0, 0] - X[0, 1])
    dy = np.abs(Y[0, 0] - Y[1, 0])

    # INITIALIZATION
    Ux = np.zeros((m, n, nt))
    Vy = np.zeros((m, n, nt))

    U = np.reshape(D[0:n * m, :], (m, n, nt), order='F')
    V = np.reshape(D[n * m:2 * n * m, :], (m, n, nt), order='F')

    # GRADIENTS
    Ux[:, 1:-1, :] = (U[:, 2:, :] - U[:, 0:-2, :]) / (2 * dx)
    Ux[:, 0, :] = (-3 * U[:, 0, :] + 4 * U[:, 1, :] - U[:, 2, :]) / (2 * dx)
    Ux[:, -1, :] = (3 * U[:, -1, :] - 4 * U[:, -2, :] + U[:, -3, :]) / (2 * dx)

    Vy[1:-1, :, :] = (V[2:, :, :] - V[0:-2, :, :]) / (2 * dy)
    Vy[0, :, :] = (-3 * V[0, :, :] + 4 * V[1, :, :] - V[2, :, :]) / (2 * dy)
    Vy[-1, :, :] = (3 * V[-1, :, :] - 4 * V[-2, :, :] + V[-3, :, :]) / (2 * dy)

    # CORRECT FOR POINTS NEAR MASK
    iR, iT, iL, iB = get_mask_boundaries(grid['B'])

    Ux[iR[0], iR[1], :] = (-3 * U[iR[0], iR[1], :] + 4 * U[iR[0], iR[1] + 1, :] - U[iR[0], iR[1] + 2, :]) / (2 * dx)
    Ux[iL[0], iL[1], :] = (3 * U[iL[0], iL[1], :] - 4 * U[iL[0], iL[1] - 1, :] + U[:iL[0], iL[1] - 2, :]) / (2 * dx)

    Vy[iT[0], iT[1], :] = (-3 * V[iT[0], iT[1], :] + 4 * V[iT[0] + 1, iT[1], :] - V[iT[0] + 2, iT[1], :]) / (2 * dy)
    Vy[iB[0], iB[1], :] = (3 * V[iB[0], iB[1], :] - 4 * V[iB[0] - 1, iB[1], :] + V[iB[0] - 2, iB[1], :]) / (2 * dy)

    imask = np.where(grid['B'] == 1)
    Ux[imask[0], imask[1], :] = 0
    Vy[imask[0], imask[1], :] = 0

    # DIVERGENCE
    Div = np.reshape(Ux + Vy, (n * m, nt), order='F')

    return Div

def diff_2nd_2D(grid,D):

    # PARAMETERS
    X = grid['X']
    Y = grid['Y']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(D)[1]

    dx = np.abs(X[0, 0] - X[0, 1])
    dy = np.abs(Y[0, 0] - Y[1, 0])

    # INITIALIZATION
    Uxx = np.zeros((m,n,nt))
    Uyy = np.zeros((m,n,nt))
    Uxy = np.zeros((m,n,nt))

    Vxx = np.zeros((m, n, nt))
    Vyy = np.zeros((m, n, nt))
    Vxy = np.zeros((m, n, nt))

    U = np.reshape(D[0:n * m, :], (m, n, nt), order='F')
    V = np.reshape(D[n * m:2 * n * m, :], (m, n, nt), order='F')

    # GRADIENTS
    Uxx[:, 1:-1, :] = (U[:, 2:, :] - 2 * U[:, 1:-1, :] + U[:, 0:-2, :]) / dx ** 2
    Uxx[:, 0, :] = (2 * U[:, 0, :] - 5 * U[:, 1, :] + 4 * U[:, 2, :] - U[:, 3, :]) / dx ** 3
    Uxx[:, -1, :] = (2 * U[:, -1, :] - 5 * U[:, -2, :] + 4 * U[:, -3, :] - U[:, -4, :]) / dx ** 3

    Uyy[1:-1, :, :] = (U[2:, :, :] - 2 * U[1:-1, :, :] + U[0:-2, :, :]) / dy ** 2
    Uyy[0, :, :] = (2 * U[0, :, :] - 5 * U[1, :, :] + 4 * U[2, :, :] - U[3, :, :]) / dy ** 3
    Uyy[-1, :, :] = (2 * U[-1, :, :] - 5 * U[-2, :, :] + 4 * U[-3, :, :] - U[-4, :, :]) / dy ** 3

    Uxy[1:-1, 1:-1, :] = (U[2:, 2:, :] - U[0:-2, 2:, :] - U[2:, 0:-2, :] + U[0:-2, 0:-2, :]) / (4 * dx * dy)
    Uxy[0, 1:-1, :] = (U[1, 2:, :] - U[0, 2:, :] - U[1, 0:-2, :] + U[0, 0:-2, :]) / (2 * dx * dy)
    Uxy[-1, 1:-1, :] = (U[-1, 2:, :] - U[-2, 2:, :] - U[-1, 0:-2, :] + U[-2, 0:-2, :]) / (2 * dx * dy)
    Uxy[1:-1, 0, :] = (U[2, 0, :] - U[0:-2, 0, :] - U[2, 1, :] + U[0:-2, 1, :]) / (2 * dx * dy)
    Uxy[1:-1, -1, :] = (U[2, -1, :] - U[0:-2, -1, :] - U[2, -2, :] + U[0:-2, -2, :]) / (2 * dx * dy)

    Vxx[:, 1:-1, :] = (V[:, 2:, :] - 2 * V[:, 1:-1, :] + V[:, 0:-2, :]) / dx ** 2
    Vxx[:, 0, :] = (2 * V[:, 0, :] - 5 * V[:, 1, :] + 4 * V[:, 2, :] - V[:, 3, :]) / dx ** 3
    Vxx[:, -1, :] = (2 * V[:, -1, :] - 5 * V[:, -2, :] + 4 * V[:, -3, :] - V[:, -4, :]) / dx ** 3

    Vyy[1:-1, :, :] = (V[2:, :, :] - 2 * V[1:-1, :, :] + V[0:-2, :, :]) / dy ** 2
    Vyy[0, :, :] = (2 * V[0, :, :] - 5 * V[1, :, :] + 4 * V[2, :, :] - V[3, :, :]) / dy ** 3
    Vyy[-1, :, :] = (2 * V[-1, :, :] - 5 * V[-2, :, :] + 4 * V[-3, :, :] - V[-4, :, :]) / dy ** 3

    Vxy[1:-1, 1:-1, :] = (V[2:, 2:, :] - V[0:-2, 2:, :] - V[2:, 0:-2, :] + V[0:-2, 0:-2, :]) / (4 * dx * dy)
    Vxy[0, 1:-1, :] = (V[1, 2:, :] - V[0, 2:, :] - V[1, 0:-2, :] + V[0, 0:-2, :]) / (2 * dx * dy)
    Vxy[-1, 1:-1, :] = (V[-1, 2:, :] - V[-2, 2:, :] - V[-1, 0:-2, :] + V[-2, 0:-2, :]) / (2 * dx * dy)
    Vxy[1:-1, 0, :] = (V[2, 0, :] - V[0:-2, 0, :] - V[2, 1, :] + V[0:-2, 1, :]) / (2 * dx * dy)
    Vxy[1:-1, -1, :] = (V[2, -1, :] - V[0:-2, -1, :] - V[2, -2, :] + V[0:-2, -2, :]) / (2 * dx * dy)

    # CORRECT FOR POINTS NEAR MASK
    iR, iT, iL, iB = get_mask_boundaries(grid['B'])

    Uxx[iR[0], iR[1], :] = (2 * U[iR[0], iR[1], :] - 5 * U[iR[0], iR[1] + 1, :] + 4 * U[iR[0], iR[1] + 2, :] - U[iR[0], iR[1] + 3, :]) / dx ** 3
    Uxx[iL[0], iL[1], :] = (2 * U[iL[0], iL[1], :] - 5 * U[iL[0], iL[1] - 1, :] + 4 * U[iL[0], iL[1] - 2, :] - U[iL[0], iL[1] - 3, :]) / dx ** 3
    Vxx[iR[0], iR[1], :] = (2 * V[iR[0], iR[1], :] - 5 * V[iR[0], iR[1] + 1, :] + 4 * V[iR[0], iR[1] + 2, :] - V[iR[0], iR[1] + 3, :]) / dx ** 3
    Vxx[iL[0], iL[1], :] = (2 * V[iL[0], iL[1], :] - 5 * V[iL[0], iL[1] - 1, :] + 4 * V[iL[0], iL[1] - 2, :] - V[iL[0], iL[1] - 3, :]) / dx ** 3

    Uyy[iT[0], iT[1], :] = (2 * U[iT[0], iT[1], :] - 5 * U[iT[0] + 1, iT[1], :] + 4 * U[iT[0] + 2, iT[1], :] - U[iT[0] + 3, iT[1], :]) / dy ** 3
    Uyy[iB[0], iB[1], :] = (2 * U[iB[0], iB[1], :] - 5 * U[iB[0] - 1, iB[1], :] + 4 * U[iB[0] - 2, iB[1], :] - U[iB[0] - 3, iB[1], :]) / dy ** 3
    Vyy[iT[0], iT[1], :] = (2 * V[iT[0], iT[1], :] - 5 * V[iT[0] + 1, iT[1], :] + 4 * V[iT[0] + 2, iT[1], :] - V[iT[0] + 3, iT[1], :]) / dy ** 3
    Vyy[iB[0], iB[1], :] = (2 * V[iB[0], iB[1], :] - 5 * V[iB[0] - 1, iB[1], :] + 4 * V[iB[0] - 2, iB[1], :] - V[iB[0] - 3, iB[1], :]) / dy ** 3

    imask = np.where(grid['B'] == 1)
    Uxx[imask[0], imask[1], :] = 0
    Uyy[imask[0], imask[1], :] = 0
    Vxx[imask[0], imask[1], :] = 0
    Vyy[imask[0], imask[1], :] = 0
    Vxy[imask[0], imask[1], :] = 0
    Uxy[imask[0], imask[1], :] = 0

    # CORRECTION FOR VXY AND UXY REQUIRED

    # RESHAPE
    Dxx = np.concatenate((np.reshape(Uxx,(n*m,nt),order='F'),np.reshape(Vxx,(n*m,nt),order='F')),axis=0)
    Dyy = np.concatenate((np.reshape(Uyy,(n*m,nt),order='F'),np.reshape(Vyy,(n*m,nt),order='F')),axis=0)
    Dxy = np.concatenate((np.reshape(Uxy,(n*m,nt),order='F'),np.reshape(Vxy,(n*m,nt),order='F')),axis=0)

    return Dxx, Dyy, Dxy

def diff_1st_1D(grid,D):

    # PARAMETERS AND INITIALIZATION
    X = grid['X']
    Y = grid['Y']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(D)[1]

    dx = np.abs(X[0, 1] - X[0, 0])
    dy = np.abs(Y[0, 0] - Y[1, 0])

    U = np.reshape(D[0:m * n,:], (m, n, nt), order='F')

    Ux, Uy = np.zeros((2, m, n, nt))

    # GRADIENTS
    Ux[:, 1:-1, :] = (U[:, 2:, :] - U[:, 0:-2, :]) / (2 * dx)
    Ux[:, 0, :] = (-3 * U[:, 0, :] + 4 * U[:, 1, :] - U[:, 2, :]) / (2 * dx)
    Ux[:, -1, :] = (3 * U[:, -1, :] - 4 * U[:, -2, :] + U[:, -3, :]) / (2 * dx)

    Uy[1:-1, :, :] = (U[2:, :, :] - U[0:-2, :, :]) / (2 * dy)
    Uy[0, :, :] = (-3 * U[0, :, :] + 4 * U[1, :, :] - U[2, :, :]) / (2 * dy)
    Uy[-1, :, :] = (3 * U[-1, :, :] - 4 * U[-2, :, :] + U[-3, :, :]) / (2 * dy)

    # CORRECT FOR POINTS NEAR MASK
    iR, iT, iL, iB = get_mask_boundaries(grid['B'])

    Ux[iR[0], iR[1], :] = (-3 * U[iR[0], iR[1], :] + 4 * U[iR[0], iR[1] + 1, :] - U[iR[0], iR[1] + 2, :]) / (2 * dx)
    Ux[iL[0], iL[1], :] = (3 * U[iL[0], iL[1], :] - 4 * U[iL[0], iL[1] - 1, :] + U[iL[0], iL[1] - 2, :]) / (2 * dx)

    Uy[iT[0], iT[1], :] = (-3 * U[iT[0], iT[1], :] + 4 * U[iT[0] + 1, iT[1], :] - U[iT[0] + 2, iT[1], :]) / (2 * dy)
    Uy[iB[0], iB[1], :] = (3 * U[iB[0], iB[1], :] - 4 * U[iB[0] - 1, iB[1], :] + U[iB[0] - 2, iB[1], :]) / (2 * dy)

    imask = np.where(grid['B'] == 1)
    Ux[imask[0], imask[1], :] = 0
    Uy[imask[0], imask[1], :] = 0

    # RESHAPE
    Dx = np.reshape(Ux, (m * n, nt), order='F')
    Dy = np.reshape(Uy, (m * n, nt), order='F')

    return Dx, Dy