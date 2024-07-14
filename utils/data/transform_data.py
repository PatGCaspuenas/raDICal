# PACKAGES
import numpy as np
from sklearn.model_selection import train_test_split

def raw2CNNAE(grid, D, flag_split=0, flag_control=0, u=0):

    # FLOW PARAMETERS
    X = grid['X']
    Y = grid['Y']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(D)[1]

    if np.shape(D)[0] == n * m:
        k = 1
    elif np.shape(D)[0] == 2 * n * m:
        k = 2
    else:
        k = 3

    # CONTROL PARAMETERS
    if flag_control:
        nc = np.shape(u)[1]

    # RESHAPE FLOW (control is already reshaped)
    U = np.zeros((nt, n, m, k))
    for i in range(k):
        U[:, :, :, i] = np.reshape(D[( (n * m)*i ):( (n * m)*(i + 1) ), :], (m, n, nt), order='F').T

    # SPLIT AND RETURN
    if flag_split and flag_control:
        X_train, X_test, u_train, u_test = train_test_split(U, u, test_size=0.3, shuffle=False)
        return X_train, X_test, u_train, u_test
    elif flag_split and ~flag_control:
        X_train, X_test = train_test_split(U, test_size=0.3, shuffle=False)
        return X_train, X_test, 0, 0
    else:
        return U

def CNNAE2raw(U):

    # PARAMETERS
    m = np.shape(U)[1]
    n = np.shape(U)[2]
    nt = np.shape(U)[0]
    k = np.shape(U)[3]

    # INITIALIZATION
    u = np.array(U).T # (k, m, n, nt)
    D = np.zeros((k*m*n, nt))

    # RESHAPE
    for i in range(k):

        D[( (n * m)*i ):( (n * m)*(i + 1) ), :] = np.reshape(u[i, :, :, :], (m*n, nt), order='F')

    return D

def get_control_vector(flow, flag_flow, flag_control):

    if flag_flow=='FP':
        nt = np.shape(flow['U'])[1]
        if flag_control:
            if 'vF' in flow:
                u = np.concatenate((flow['vF'], flow['vT'], flow['vB']), axis=1)
            else:
                u = np.zeros((nt, 3))
        else:
            u = np.zeros((nt, 3))
    else:
        u = 0

    return u

def get_mask_boundaries(Mask):

    B = np.zeros_like(Mask)
    B[np.isnan(Mask)] = 0
    B[Mask==1] = 1

    # RIGHT BOUNDARY
    Baux = np.zeros_like(B)
    Baux[:, 1:] = B[:, :-1]
    i_right = np.where((Baux-B)==1)

    # TOP BOUNDARY
    Baux = np.zeros_like(B)
    Baux[1:, :] = B[:-1, :]
    i_top = np.where((Baux - B) == 1)

    # LEFT BOUNDARY
    Baux = np.zeros_like(B)
    Baux[:, :-1] = B[:, 1:]
    i_left = np.where((Baux - B) == 1)

    # BOTTOM BOUNDARY
    Baux = np.zeros_like(B)
    Baux[:-1, :] = B[1:, :]
    i_bottom = np.where((Baux - B) == 1)

    return i_right, i_top, i_left, i_bottom

def raw2dyn(t, z, params, flag_control, u = np.zeros((1,1)), flag_train = 1):

    dt = params['dyn']['dt']
    dt_lat = params['dyn']['dt_lat']
    tol = 1e-5

    o = params['dyn']['o']
    n_p = params['dyn']['np']
    nt_pred = params['dyn']['nt_pred']
    d = params['dyn']['d']

    nr = np.shape(z)[1]
    nc = np.shape(u)[1] if flag_control else 0

    if flag_train:
        t_train, t_val = train_test_split(t, test_size=0.2, shuffle=False)
        flags = ['train','val']
    else:
        flags = ['test']

    for flag in flags:

        T = {'TDL_lat': [], 'pred': []}

        stop = 0

        if flag=='train':
            tw = t_train[0] + dt_lat * (d - 1)
            tf = t_train[-1]
        elif flag == 'val':
            tw = t_val[0] + dt_lat * (d - 1)
            tf = t_val[-1]
        elif flag=='test':
            tw = t[0] + dt_lat * (d - 1)
            tf = t[-1]

        while (not stop):

            T['TDL_lat'].append(np.linspace(tw - dt_lat * (d - 1), tw, d  * int( dt_lat / dt ) -  int( dt_lat / dt  - 1)))
            if flag=='test':
                T['pred'].append(np.linspace(tw + dt, tw + dt * nt_pred, nt_pred))
            else:
                T['pred'].append(np.linspace(tw + dt, tw + dt * n_p, n_p))

            tw = tw + dt * o

            if flag == 'test':
                if (tw + dt * nt_pred) >= tf:
                    stop = 1
            else:
                if (tw + dt * n_p) >= tf:
                    stop = 1

        nW = len(T['pred'])
        Zx = np.zeros((nW, d  * int( dt_lat / dt ) -  int( dt_lat / dt  - 1), nr))
        Zy = np.zeros((nW, nt_pred, nr)) if (flag=='test') else np.zeros((nW, n_p, nr))

        if flag_control:
            Ux = np.zeros((nW, d  * int( dt_lat / dt ) -  int( dt_lat / dt  - 1), nc))
            Uy = np.zeros((nW, nt_pred, nc)) if (flag=='test') else np.zeros((nW, n_p, nc))

        for i in range(nW):

            it_TDL_lat = np.arange(np.where(np.abs(T['TDL_lat'][i][0]-t) < tol)[0], np.where(np.abs(T['TDL_lat'][i][-1]-t) < tol)[0] + 1)
            it_pred = np.arange(np.where(np.abs(T['pred'][i][0]-t) < tol)[0], np.where(np.abs(T['pred'][i][-1]-t) < tol)[0] + 1)

            Zx[i, :, :] = z[it_TDL_lat, :]
            Zy[i, :, :] = z[it_pred, :]

            if flag_control:
                Ux[i, :, :] = u[it_TDL_lat, :]
                Uy[i, :, :] = u[it_pred, :]

        if flag=='train':
            zx_train, zy_train = Zx, Zy
            if flag_control:
                ux_train, uy_train = Ux, Uy
        elif flag=='val':
            zx_val, zy_val = Zx, Zy
            if flag_control:
                ux_val, uy_val = Ux, Uy
        elif flag == 'test':
            zx, zy = Zx, Zy
            if flag_control:
                ux, uy = Ux, Uy

    if flag_train:
        if flag_control:
            return zx_train, zy_train, zx_val, zy_val, ux_train, uy_train, ux_val, uy_val
        else:
            return zx_train, zy_train, zx_val, zy_val
    else:
        if flag_control:
            return zx, zy, ux, uy, T
        else:
            return zx, zy, T

def z_window2concat(X, n_p):

    nW, nt, nr = np.shape(X[:, 0:n_p, :])

    X = np.reshape(X[:, 0:n_p, :], (nW * nt, nr))

    return X

def flow2window(Ddt, t, t_pred):

    nW = len(t_pred)
    ntpred = len(t_pred[0])
    tol = 1e-5

    nv, nt = np.shape(Ddt)

    Dw = np.zeros((nW, nv, ntpred))

    for w in range(nW):

        it0 = np.where(np.abs(t - t_pred[w][0]) < tol)[0][0]
        itf = np.where(np.abs(t - t_pred[w][-1]) < tol)[0][0] + 1

        Dw[w, :, :] = Ddt[:, it0:itf]

    return Dw

def window2flow(Dw):

    nW, nv, nt = np.shape(Dw)

    for w in range(nW):
        if w == 0:
            Ddt = Dw[0,:,:]
        else:
            Ddt = np.concatenate((Ddt, Dw[w,:,:]), axis=1)

    return Ddt