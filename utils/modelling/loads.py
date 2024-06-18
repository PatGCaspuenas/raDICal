import numpy as np

from utils.modelling.differentiation import diff_time
from utils.modelling.differentiation import diff_1st_2D

def get_aerodynamic_coefs(grid, Re, t, D, P):

    # PARAMETERS
    X = grid['X']
    Y = grid['Y']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(D)[1]

    dx = np.abs(X[0,0] - X[0,1])
    dy = np.abs(Y[0,0] - Y[1,0])

    # DIFFERENTIATION
    dDdt = diff_time(D, t)
    Dx, Dy = diff_1st_2D(grid, D)

    # DECOUPLE U AND V COMPONENTS
    U = np.reshape(D[0:m * n, :], (m, n, nt), order='F')
    V = np.reshape(D[m * n:, :], (m, n, nt), order='F')

    Ux = np.reshape(Dx[0:m * n, :], (m, n, nt), order='F')
    Vx = np.reshape(Dx[m * n:, :], (m, n, nt), order='F')
    Uy = np.reshape(Dy[0:m * n, :], (m, n, nt), order='F')
    Vy = np.reshape(Dy[m * n:, :], (m, n, nt), order='F')

    dUdt = np.reshape(dDdt[0:m * n, :], (m, n, nt), order='F')
    dVdt = np.reshape(dDdt[m * n:, :], (m, n, nt), order='F')

    P = np.reshape(P, (m, n, nt), order='F')

    del D, dDdt, Dx, Dy

    pressure_x = dy * np.sum(P[:, 0, :], axis=0) - dy * np.sum(P[:, -1, :], axis=0)
    pressure_y = dx * np.sum(P[0, :, :], axis=0) - dx * np.sum(P[-1, :, :], axis=0)

    convection_x = dy * np.sum(U[:, 0, :] ** 2, axis=0) - dy * np.sum(U[:, -1, :] ** 2, axis=0) + \
                   dx * np.sum(np.multiply(U[0, :, :], V[0, :, :]), axis=0) - dx * np.sum(np.multiply(U[-1, :, :], V[-1, :, :]), axis=0)
    convection_y = dy * np.sum(np.multiply(U[:, 0, :], V[:, 0, :]), axis=0) - dy * np.sum(np.multiply(U[:, -1, :], V[:, -1, :]), axis=0) + \
                   dx * np.sum(V[0, :, :] ** 2, axis=0) - dx * np.sum(V[-1, :, :] ** 2, axis=0)

    acceleration_x = dx * dy * np.sum(dUdt, axis=[0,1])
    acceleration_y = dx * dy * np.sum(dVdt, axis=[0,1])

    viscous_x = dy / Re * np.sum(Ux[:, -1, :] ** 2, axis=0) - dy / Re * np.sum(Ux[:, 0, :] ** 2, axis=0) + \
                dx / Re * np.sum(Uy[-1, :, :] + Vx[-1, :, :], axis=0) - dx / Re * np.sum(Uy[0, :, :] + Vx[0, :, :], axis=0)
    viscous_y = dy / Re * np.sum(Uy[:, -1, :] + Vx[:, -1, :], axis=0) - dy / Re * np.sum(Uy[:, 0, :] + Vx[:, 0, :], axis=0) + \
                dx / Re * np.sum(Vy[-1, :, :] ** 2, axis=0) - dx / Re * np.sum(Vy[0, :, :] ** 2, axis=0)

    CL = (pressure_y + convection_y + acceleration_y + viscous_y) / 2
    CD = (pressure_x + convection_x + acceleration_x + viscous_x) / 2

    return CL, CD