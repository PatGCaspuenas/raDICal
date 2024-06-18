def get_p():
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    from utils.modelling.differentiation import diff_time, diff_1st_2D, diff_1st_1D
    from utils.modelling.pressure import get_pgrad, p_integrator

    path_flow = r'F:\AEs_wControl\DATA\FPp_10k_13k.h5'
    path_grid = r'F:\AEs_wControl\DATA\FP_grid.h5'

    flow = {}
    with h5py.File(path_flow, 'r') as f:
        for i in f.keys():
            flow[i] = f[i][()]

    grid = {}
    with h5py.File(path_grid, 'r') as f:
        for i in f.keys():
            grid[i] = f[i][()]

    t = flow['t']

    X = grid['X']
    Y = grid['Y']
    m = np.shape(grid['X'])[0]
    n = np.shape(grid['X'])[1]
    dx = np.abs(X[0, 0] - X[0, 1])
    dy = np.abs(Y[0, 0] - Y[1, 0])

    nt = np.shape(t)[0]
    Re = flow['Re']

    U = np.reshape(flow['U'], (m, n, nt), order='F')
    V = np.reshape(flow['V'], (m, n, nt), order='F')
    U, V = np.flip(U, axis=0), np.flip(V, axis=0)
    P_ref = np.reshape(flow['P'], (m, n, nt), order='F')
    P_ref = np.flip(P_ref, axis=0)

    D = np.concatenate((np.reshape(U, (m * n, nt), order='F'), np.reshape(V, (m * n, nt), order='F')), axis=0)
    # D = np.concatenate((flow['U'], flow['V']), axis=0)
    dDdt_ref = np.concatenate((flow['dUdt'], flow['dVdt']), axis=0)
    dDdt = diff_time(D, t)
    err_dDdt = np.sum(np.abs(dDdt_ref - dDdt))
    Px, Py = diff_1st_1D(grid, flow['P'])
    del flow

    Dx, Dy = diff_1st_2D(grid, D)

    Ux = np.reshape(Dx[0:m * n, :], (m, n, nt), order='F')
    Uy = np.reshape(Dy[0:m * n, :], (m, n, nt), order='F')
    Vx = np.reshape(Dx[m * n:, :], (m, n, nt), order='F')
    Vy = np.reshape(Dy[m * n:, :], (m, n, nt), order='F')

    # GET STRESSES ON OUTFLOW BOUNDARY
    taux = np.sum(1 / Re * (Ux[:, -1, :] ** 2) * dy, axis=0)
    tauy = np.sum(1 / Re * (Vx[:, -1, :] + Uy[:, -1]) * dy, axis=0)

    print('mean taux:' + str(np.mean(taux)) + ', std taux:' + str(np.std(taux)))
    print('mean tauy:' + str(np.mean(tauy)) + ', std tauy:' + str(np.std(tauy)))

    # GET P GRADIENT
    DP = get_pgrad(grid, D, dDdt, Re)
    P = p_integrator(grid, DP, P_ref[:, :, 0], P_ref[50, 20, :])
    P = np.reshape(P, (m, n, nt), order='F')

    # CHECK INFLOW BOUNDARY
    fig, ax = plt.subplots(1, 1)
    for i in range(m):
        ax.plot(t, Y[i, 0] + (U[i, 0, :] - 1))
    ax.grid()
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$y/D + (u_i - U_{\infty})$')

    fig, ax = plt.subplots(1, 1)
    for i in range(m):
        ax.plot(t, Y[i, 0] + (V[i, 0, :]))
    ax.grid()
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$y/D + v_i$')

    fig, ax = plt.subplots(1, 1)
    for i in range(m):
        ax.plot(t, Y[i, 0] + (P_ref[i, 0, :]))
    ax.grid()
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$y/D + (P_i - P_{\infty})$')

    # CHECK BOTTOM BOUNDARY
    fig, ax = plt.subplots(1, 1)
    for i in range(n):
        ax.plot(t, X[0, i] + (U[0, i, :] - 1))
    ax.grid()
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$x/D + (u_b - U_{\infty})$')

    fig, ax = plt.subplots(1, 1)
    for i in range(n):
        ax.plot(t, X[0, i] + (V[0, i, :]))
    ax.grid()
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$x/D + v_b$')

    fig, ax = plt.subplots(1, 1)
    for i in range(n):
        ax.plot(t, X[0, i] + (P_ref[0, i, :]))
    ax.grid()
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$x/D + (P_b - P_{\infty})$')

    # CHECK TOP BOUNDARY
    fig, ax = plt.subplots(1, 1)
    for i in range(n):
        ax.plot(t, X[-1, i] + (U[-1, i, :] - 1))
    ax.grid()
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$x/D + (u_t - U_{\infty})$')

    fig, ax = plt.subplots(1, 1)
    for i in range(n):
        ax.plot(t, X[-1, i] + (V[-1, i, :]))
    ax.grid()
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$x/D + v_t$')

    fig, ax = plt.subplots(1, 1)
    for i in range(n):
        ax.plot(t, X[-1, i] + (P_ref[-1, i, :]))
    ax.grid()
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$x/D + (P_t - P_{\infty})$')

    fig, ax = plt.subplots(1, 1)

    ax.plot(X[-1, :], np.mean(P_ref[-1, :, :], axis=1), '-', color='r', label='mean P_t')
    ax.plot(X[-1, :], np.mean(U[-1, :, :] - 1, axis=1), '-', color='m', label='mean U_t')
    ax.plot(X[-1, :], np.mean(V[-1, :, :], axis=1), '-', color='b', label='mean V_t')
    ax.plot(X[-1, :], np.std(P_ref[-1, :, :], axis=1), '--', color='r', label='std P_t')
    ax.plot(X[-1, :], np.std(U[-1, :, :] - 1, axis=1), '--', color='m', label='std U_t')
    ax.plot(X[-1, :], np.std(V[-1, :, :], axis=1), '--', color='b', label='std V_t')

    ax.plot(X[0, :], np.mean(P_ref[0, :, :], axis=1), '-', color='k', label='mean P_b')
    ax.plot(X[0, :], np.mean(U[0, :, :] - 1, axis=1), '-', color='y', label='mean U_b')
    ax.plot(X[0, :], np.mean(V[0, :, :], axis=1), '-', color='c', label='mean V_b')
    ax.plot(X[0, :], np.std(P_ref[0, :, :], axis=1), '--', color='k', label='std P_b')
    ax.plot(X[0, :], np.std(U[0, :, :] - 1, axis=1), '--', color='y', label='std U_b')
    ax.plot(X[0, :], np.std(V[0, :, :], axis=1), '--', color='c', label='std V_b')

    ax.grid()
    ax.legend(loc='upper right')
    ax.set_xlabel(r'$x/D$')
    ax.set_ylabel(r'$\varepsilon$')

    fig, ax = plt.subplots(1, 1)

    ax.plot(Y[:, 0], np.mean(P_ref[:, 0, :], axis=1), '-', color='k', label='mean P_i')
    ax.plot(Y[:, 0], np.mean(U[:, 0, :] - 1, axis=1), '-', color='y', label='mean U_i')
    ax.plot(Y[:, 0], np.mean(V[:, 0, :], axis=1), '-', color='c', label='mean V_i')
    ax.plot(Y[:, 0], np.std(P_ref[:, 0, :], axis=1), '--', color='k', label='std P_i')
    ax.plot(Y[:, 0], np.std(U[:, 0, :] - 1, axis=1), '--', color='y', label='std U_i')
    ax.plot(Y[:, 0], np.std(V[:, 0, :], axis=1), '--', color='c', label='std V_i')

    ax.grid()
    ax.legend(loc='upper right')
    ax.set_xlabel(r'$y/D$')
    ax.set_ylabel(r'$\varepsilon$')

    a = 0

    # CHECK AERODYNAMIC FORCES