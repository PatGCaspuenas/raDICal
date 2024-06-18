import numpy as np
from scipy.interpolate import RegularGridInterpolator
from utils.modelling.differentiation import *

def get_pgrad(grid, D, dDdt, Re):

    # PARAMETERS
    X = grid['X']
    Y = grid['Y']

    m = X.shape[0]
    n = X.shape[1]

    U = D[0:m*n, :]
    V = D[m*n:2*m*n, :]

    # GET GRADIENT
    Dx, Dy = diff_1st_2D(grid, D)
    Ux = Dx[0:m*n, :]
    Uy = Dy[0:m*n, :]
    Vx = Dx[m*n:2*m*n, :]
    Vy = Dy[m*n:2*m*n, :]

    # GET LAPLACIAN
    D2D = get_laplacian_2D(grid, D)

    # GET P GRADIENT
    DXD = np.concatenate((np.multiply(U,Ux) + np.multiply(V,Uy), np.multiply(U,Vx) + np.multiply(V,Vy)), axis=0)

    DP = 1/Re*D2D - dDdt - DXD

    return DP

def p_integrator(grid, Dp, p_init, p_bc):

    # PARAMETERS
    X = grid['X']
    Y = grid['Y']
    B = grid['B']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(Dp)[1]

    dx = np.abs(X[0, 0] - X[0, 1])
    dy = np.abs(Y[0, 0] - Y[1, 0])

    tol = 1e-4
    lambda0 = 0.15

    # CREATE MASK
    M = np.zeros_like(B)
    M[np.isnan(B)] = 1
    M[B==1] = 0

    mask = np.ones((m + 2, n + 2))
    mask[1:-1, 1:-1] = M
    mask1 = np.multiply(np.roll(mask, 1, 1), mask)
    mask2 = np.multiply(np.roll(mask, -1, 1), mask)
    mask3 = np.multiply(np.roll(mask, 1, 0), mask)
    mask4 = np.multiply(np.roll(mask, -1, 0), mask)

    p = np.zeros((m, n, nt))
    for t in range(nt):
        # RESHAPE PRESSURE GRADIENTS
        px = np.reshape(Dp[0:m * n, t], (m, n), order='F')
        py = np.reshape(Dp[m * n:2 * m * n, t], (m, n), order='F')

        # INITIALIZE ITERATION
        P = np.zeros((m + 2, n + 2))
        P[1:-1, 1:-1] = p_init

        P[0 ,  :] = P[1 ,  :]
        P[-1,  :] = P[-2,  :]
        P[: ,  0] = P[: ,  1]
        P[: , -1] = P[: , -2]

        PX = px[:, 1:] / 2 + px[:, :-1] / 2

        PX = np.concatenate((PX[0,:].reshape((1,n-1)), PX, PX[-1, :].reshape((1,n-1))),axis=0)
        PX1 = np.concatenate((0 * PX[:, 0:2].reshape((m+2,2)), PX, 0 * PX[:, 0].reshape((m+2,1))), axis=1)
        PX2 = np.concatenate((0 * PX[:, 0].reshape((m+2,1)), PX, 0 * PX[:, 0:2].reshape((m+2,2))), axis=1)

        PY = py[1:, :] / 2 + py[:-1, :] / 2

        PY = np.concatenate((PY[:, 0].reshape((m-1,1)), PY, PY[:, -1].reshape((m-1,1))), axis=1)
        PY1 = np.concatenate((0 * PY[0:2, :].reshape((2,n+2)), PY, 0 * PY[0, :].reshape((1,n+2))), axis=0)
        PY2 = np.concatenate((0 * PY[0, :].reshape((1,n+2)), PY, 0 * PY[0:2, :].reshape((2,n+2))), axis=0)

        for iCount in range(100000):
            Lambda = lambda0 / np.sqrt(iCount / 50000 + 1) + 0.1
            PD = (+dx * PX1 - np.multiply((P - np.roll(P, 1, 1)), mask1) + \
                  -dx * PX2 - np.multiply((P - np.roll(P,      -1, 1)), mask2) + \
                  +dy * PY1 - np.multiply((P - np.roll(P, 1, 0)), mask3) + \
                  -dy * PY2 - np.multiply((P - np.roll(P,      -1, 0)), mask4))

            P = P + Lambda * PD
            P = np.multiply(P, mask)

            P[0 ,  :] = P[1 ,  :]
            P[-1,  :] = P[-2,  :]
            P[: ,  0] = P[: ,  1]
            P[: , -1] = P[: , -2]


            if np.mean(np.abs(PD[1:-1, 1:-1])) < tol:
                break

        p[:, :, t] = P[1:-1, 1:-1]

        # IMPOSE BC
        p[:, :, t] = p[:, :, t] + (p_bc[t] - p[50,20, t])

        p_init = p[:, :, t]

    imask = np.where(B==1)
    p[imask[0], imask[1], :] = 0
    p = np.reshape(p, (m * n, nt), order='F')
    return p

    # Xq, Yq = np.meshgrid(np.arange(1, n + 1, 1), np.arange(1.5, m + 0.5, 1))
    #
    # for t in range(nt):
    #     if t == 0:
    #         PY = RegularGridInterpolator((np.arange(1, m + 1),np.arange(1, n + 1)), py[:,:,0], method='cubic')((Yq,Xq))
    #     else:
    #         PY = np.dstack((PY, RegularGridInterpolator((np.arange(1, m + 1),np.arange(1, n + 1)), py[:,:,t], method='cubic')((Yq,Xq))))

    # Xq, Yq = np.meshgrid(np.arange(1.5, n + 0.5, 1), np.arange(1, m + 1, 1))
    #
    # for t in range(nt):
    #     if t == 0:
    #         PX = RegularGridInterpolator((np.arange(1, m + 1),np.arange(1, n + 1)), px[:,:,0], method='cubic')((Yq,Xq))
    #     else:
    #         PX = np.dstack((PX, RegularGridInterpolator((np.arange(1, m + 1),np.arange(1, n + 1)), px[:,:,t], method='cubic')((Yq,Xq))))

# COMMENT SLIDES -> ABOUT LATENT VECTORS (OSCILLATIONS), ABOUT THE ROLE OF CONTROL, ABOUT THE MODES, ABOUT THE LACK OF RESUOLUTION NERAR CYLINDERS
# PREPARE SLIDES OF PRESSURE AND FORCES
# SAVE AGAIN P BC
# FLIP U AND V IN AXIS 0 FOR NO CONTROL CASE (AND CHECK FOR CONTROL)
# MATLAB COINCIDES BUT PYTHON NO?!

