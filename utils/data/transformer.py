# PACKAGES
import numpy as np
from sklearn.model_selection import train_test_split

def raw2CNNAE(grid, D, flag_train=0, flag_control=0, b=0):
    """
    Transforms dataset from snapshot matrix shape (NxNyK,Nt) to AE shape (Nt,Nx,Ny,K)
    :param grid: dictionary containing X,Y grids and body mask
    :param D: snapshot matrix
    :param flag_train: True if 70/30 splitting is to be done, False otherwise
    :param flag_control: True if control parameters are to be considered, False otherwise
    :param b: control vector, if required
    :return: transformed snapshot matrix and control vector
    """

    # FLOW PARAMETERS
    X = grid['X']

    N_y, N_x = np.shape(X)
    N_v, N_t = np.shape(D)
    K = N_v // (N_y * N_x)

    # CONTROL PARAMETERS
    if flag_control:
        N_c = np.shape(b)[1]

    # RESHAPE FLOW (control is already reshaped)
    U = np.zeros((N_t, N_x, N_y, K))
    for i in range(K):
        U[:, :, :, i] = np.reshape(D[( (N_y * N_x)*i ):( (N_y * N_x)*(i + 1) ), :], (N_y, N_x, N_t), order='F').T

    # SPLIT AND RETURN
    if flag_train and flag_control:
        X_train, X_test, u_train, u_test = train_test_split(U, b, test_size=0.3, shuffle=False)
        return X_train, X_test, u_train, u_test
    elif flag_train and ~flag_control:
        X_train, X_test = train_test_split(U, test_size=0.3, shuffle=False)
        return X_train, X_test, 0, 0
    else:
        return U

def CNNAE2raw(U):
    """
    Transforms dataset from AE shape (Nt,Nx,Ny,K) to snapshot matrix shape (NxNyK,Nt)
    :param U: dataset in AE shape
    :return: dataset in snapshot matrix shape
    """

    # PARAMETERS
    N_t, N_x, N_y, K = np.shape(U)

    # INITIALIZATION
    UT = np.array(U).T
    D = np.zeros((K * N_x * N_y, N_t))

    # RESHAPE
    for i in range(K):

        D[( (N_x * N_y)*i ):( (N_x * N_y)*(i + 1) ), :] = np.reshape(UT[i, :, :, :], (N_x * N_y, N_t), order='F')

    return D

def get_control_vector(flow, flag_flow, flag_control):
    """
    Reshape control vector depending on flow configuration
    :param flow: dictionary containing control coordinates, among others
    :param flag_flow: type of flow flag
    :param flag_contro: 1 if flow configuration is controlled, 0 otherwise
    :return: control vector (N_t, N_c)
    """

    if flag_flow=='FP':
        if flag_control:
            b = np.concatenate((flow['vF'], flow['vT'], flow['vB']), axis=1)
        else:
            b = 0
    elif flag_flow=='SC':
        b = 0

    return b

def get_mask_boundaries(Mask):
    """
    Retrieves 2D indices of mask at body boundary
    :param Mask: (N_y, N_x) grid containing 1s on body and 0 otherwise
    :return: indices to the right, top, left and bottom of the body boundary
    """

    B = np.zeros_like(Mask)
    B[np.isnan(Mask)] = 0
    B[Mask==1] = 1

    # RIGHT BOUNDARY
    BaBx = np.zeros_like(B)
    BaBx[:, 1:] = B[:, :-1]
    i_right = np.where((BaBx-B)==1)

    # TOP BOUNDARY
    BaBx = np.zeros_like(B)
    BaBx[1:, :] = B[:-1, :]
    i_top = np.where((BaBx - B) == 1)

    # LEFT BOUNDARY
    BaBx = np.zeros_like(B)
    BaBx[:, :-1] = B[:, 1:]
    i_left = np.where((BaBx - B) == 1)

    # BOTTOM BOUNDARY
    BaBx = np.zeros_like(B)
    BaBx[:-1, :] = B[1:, :]
    i_bottom = np.where((BaBx - B) == 1)

    return i_right, i_top, i_left, i_bottom

def raw2dyn(t, z, PARAMS, flag_control, b = np.zeros((1,1)), flag_train = 1):
    """
    Converts state vector (and control vector) into shape required by dynamical predictors (tapped delay and prediction windows)
    :param t: time vector (N_t, 1)
    :param z: state vector (N_t, N_z)
    :param PARAMS: dictionary containing parameters
    :param flag_control: 1 if control should be considered, 0 otherwise
    :param b: control vector (N_t, N_c)
    :param flag_train: 1 if train-val splitting is to be done, 0 otherwise
    :return: tapped delay line and prediction window for both state vector and control vector (if any)
    """

    # Read parameters
    DT = PARAMS['DYN']['DT'] # Delta time of the system in c.b
    tol = 1e-5

    w_o = PARAMS['DYN']['w_o'] # Offset in number of instants
    w_p = PARAMS['DYN']['w_p'] # Trained prediction window in number of instants
    w_prop = PARAMS['DYN']['w_prop'] # Propagation prediction window in number of instants
    w_d = PARAMS['DYN']['w_d'] # Tapped delay line in number of instants

    N_z = np.shape(z)[1] # Number of state coordinates
    N_c = np.shape(b)[1] if flag_control else 0 # Number of control coordinates

    # Split datasets (if needed)
    if flag_train:
        t_train, t_val = train_test_split(t, test_size=0.2, shuffle=False)
        flags_set = ['train','val']
    else:
        flags_set = ['test']

    for flag_set in flags_set:

        T = {'TDL': [], 'PW': []} # Tapped Delay Line // Prediction Window

        stop = 0

        # Determine time bounds for TDL and PW (t_w corresponds to final time of TDL, t_f is the final time of the whole dataset)
        if flag_set=='train':
            t_w = t_train[0] + DT * (w_d - 1)
            t_f = t_train[-1]
        elif flag_set == 'val':
            t_w = t_val[0] + DT * (w_d - 1)
            t_f = t_val[-1]
        elif flag_set=='test':
            t_w = t[0] + DT * (w_d - 1)
            t_f = t[-1]

        # Create time vectors for each TDL-PW
        while (not stop):

            # Update the current TDL-PW
            T['TDL'].append(np.linspace(t_w - DT * (w_d - 1), t_w, w_d))
            if flag_set=='test':
                T['PW'].append(np.linspace(t_w + DT, t_w + DT * w_prop, w_prop))
            else:
                T['PW'].append(np.linspace(t_w + DT, t_w + DT * w_p, w_p))

            # Update next TDL-PW and stop next iteration if reached t_f
            t_w = t_w + DT * w_o

            if flag_set == 'test':
                if (t_w + DT * w_prop) >= t_f:
                    stop = 1
            else:
                if (t_w + DT * w_p) >= t_f:
                    stop = 1

        # Initialize state and control vector (x is TDL (input of dynamical predictors), y is PW (output of dynamical predictors)
        N_w = len(T['PW']) # Number of windows
        Zx = np.zeros((N_w, w_d, N_z))
        Zy = np.zeros((N_w, w_prop, N_z)) if (flag_set=='test') else np.zeros((N_w, w_p, N_z))

        if flag_control:
            Bx = np.zeros((N_w, w_d, N_c))
            By = np.zeros((N_w, w_prop, N_c)) if (flag_set=='test') else np.zeros((N_w, w_p, N_c))

        # Re-order state and control vector according to time vector
        for i in range(N_w):

            # Indices
            iTDL = np.arange(np.where(np.abs(T['TDL'][i][0]-t) < tol)[0], np.where(np.abs(T['TDL'][i][-1]-t) < tol)[0] + 1)
            iPW = np.arange(np.where(np.abs(T['PW'][i][0]-t) < tol)[0], np.where(np.abs(T['PW'][i][-1]-t) < tol)[0] + 1)

            Zx[i, :, :] = z[iTDL, :]
            Zy[i, :, :] = z[iPW, :]

            if flag_control:
                Bx[i, :, :] = b[iTDL, :]
                By[i, :, :] = b[iPW, :]

        # Save different outputs depending on dataset type and control flag
        if flag_set == 'train':
            zx_train, zy_train = Zx, Zy
            if flag_control:
                bx_train, by_train = Bx, By
        elif flag_set == 'val':
            zx_val, zy_val = Zx, Zy
            if flag_control:
                bx_val, by_val = Bx, By
        elif flag_set == 'test':
            zx, zy = Zx, Zy
            if flag_control:
                bx, by = Bx, By

    # Return outputs depending on dataset type and control flag
    if flag_train:
        if flag_control:
            return zx_train, zy_train, zx_val, zy_val, bx_train, by_train, bx_val, by_val
        else:
            return zx_train, zy_train, zx_val, zy_val
    else:
        if flag_control:
            return zx, zy, bx, by, T
        else:
            return zx, zy, T

def window2zcat(X, w_pe):
    """
    Re-shapes state vector from PW approach (N_w, N_t, N_z) to concatenated format (N_w * N_t,N_z). Required for error estimation
    :param X: state vector in PW shape
    :param n_p: number of time instants in PW to consider
    :return: state vector in concatenated shape
    """

    N_w, N_t, N_z = np.shape(X[:, 0:w_pe, :])

    X = np.reshape(X[:, 0:w_pe, :], (N_w * N_t, N_z))

    return X

def flow2window(D, t, t_PW):
    """
    Re-shapes snapshot matrix (N_v, N_t) into PW format (N_w, N_v, w_prop). w_prop can also be interchanged for w_p
    :param D: snapshot matrix (N_v, N_t)
    :param t: time sequence of snapshot matrix D (N_t, 1)
    :param t_PW: time sequence of PW (N_w, w_prop)
    :return: snapshot matrix in window shape
    """

    # Parameters
    N_w = len(t_PW)
    w_prop = len(t_PW[0])
    tol = 1e-5

    N_v, N_t = np.shape(D)

    # Convert snapshot matrix into PW format
    Dw = np.zeros((N_w, N_v, w_prop))

    for w in range(N_w):

        i0 = np.where(np.abs(t - t_PW[w][0]) < tol)[0][0]
        ifin = np.where(np.abs(t - t_PW[w][-1]) < tol)[0][0] + 1

        Dw[w, :, :] = D[:, i0:ifin]

    return Dw

def window2flow(Dw):
    """
    Re-shapes snapshot matrix in PW format (N_w, N_v, w_prop) into raw format (N_v, N_t). w_prop can also be interchanged for w_p
    :param Dw: snapshot matrix in window shape
    :return: snapshot matrix in raw shape
    """

    N_w, N_v, N_t = np.shape(Dw)

    for w in range(N_w):
        if w == 0:
            D = Dw[0,:,:]
        else:
            D = np.concatenate((D, Dw[w,:,:]), axis=1)

    return D