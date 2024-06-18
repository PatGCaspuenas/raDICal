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


def raw2dyn(t, z, u, params, flag_type, flag_split = 0):

    dt = params['dyn']['dt']
    dt_lat = params['dyn']['latent']['dt']

    if (flag_type == 'NARX') or (flag_type == 'NARMAX'):
        dt_c = params['dyn']['control']['dt']
        d_c = params['dyn']['control']['d']
        d_lat = params['dyn']['latent']['d']
    else:
        dt_c = dt
        d_c = params['dyn']['d']
        d_lat = params['dyn']['d']

    o = params['dyn']['o']
    np = params['dyn']['np']

    nr = np.shape(z)[1]
    nc = np.shape(u)[1]

    if not flag_split:
        t_train, t_val = train_test_split(t, test_size=0.2, shuffle=False)
        flags = ['train','val']
    else:
        flags = ['test']

    for flag in flags:

        T = {'TDL_lat': [], 'TDL_c': [], 'pred': []}

        stop = 0

        if flag=='train':
            tw = t_train[0] + np.max([dt_lat * d_lat, dt_c * d_c])
            tf = t_train[-1]
        elif flag == 'val':
            tw = t_val[0] + np.max([dt_lat * d_lat, dt_c * d_c])
            tf = t_val[-1]
        elif flag=='test':
            tw = t[0] + np.max([dt_lat * d_lat, dt_c * d_c])
            tf = t[-1]

        while (not stop):

            T['TDL_lat'].append(np.arange(tw - dt_lat * d_lat, tw + dt_lat, dt_lat))
            T['TDL_c'].append(np.arange(tw - dt_c * d_c, tw + dt, dt))
            T['pred'].append(np.arange(tw + dt, tw + dt * (np + 1), dt))

            tw = tw + dt * o

            if (tw + dt * np) > tf:
                stop = 1

        nW = len(T['pred'])
        Zy, Zx = np.zeros((2, nW, d_lat, nr))
        Uy, Ux = np.zeros((2, nW, d_c, nc))

        for i in range(nW):

            it_TDL_lat = np.where(T['TDL_lat'][i] == t)
            it_pred = np.where(T['pred'][i] == t)
            it_TDL_c = np.where(T['TDL_c'][i] == t)

            Zx[i, :, :] = z[it_TDL_lat, :]
            Zy[i, :, :] = z[it_pred, :]

            Ux[i, :, :] = z[it_TDL_c, :]
            Uy[i, :, :] = z[it_pred, :]

        if flag=='train':
            zx_train, zy_train = Zx, Zy
            ux_train, uy_train = Ux, Uy
        elif flag=='val':
            zx_val, zy_val = Zx, Zy
            ux_val, uy_val = Ux, Uy
        elif flag == 'test':
            zx, zy = Zx, Zy
            ux, uy = Ux, Uy

    if not flag_split:
        return zx_train, zy_train, zx_val, zy_val, ux_train, uy_train, ux_val, uy_val
    else:
        return zx, zy, ux, uy