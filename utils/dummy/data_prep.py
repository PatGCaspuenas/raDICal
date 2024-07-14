

def merge_control_nocontrol():
    import random
    import numpy as np
    from utils.data.read_data import read_FP

    path_flowc = r'F:\AEs_wControl\utils\data\FPc_00k_70k.h5'
    path_flow = r'F:\AEs_wControl\utils\data\FP_14k_24k.h5'
    path_grid = r'F:\AEs_wControl\utils\data\FP_grid.h5'

    # LOAD DATA
    grid, flow = read_FP(path_grid, path_flow)
    del flow['dUdt'], flow['dVdt']
    flowc = read_FP(path_grid, path_flowc)[1]

    ntc = np.shape(flowc['U'])[1]
    nt = np.shape(flow['U'])[1]
    ic = random.sample([*range(ntc)], 7000)
    ic = np.sort(ic)
    i = random.sample([*range(nt)], 3000)
    i = np.sort(i)

    flow['t'] = flow['t'] + 7000
    flow['vF'], flow['vB'], flow['vT'] = np.zeros((3, nt, 1))

    flowm = {'Re': 150}

    vars = ['t', 'vB', 'vF', 'vT']
    for var in vars:
        flowm[var] = np.concatenate((flowc[var][ic, :], flow[var][i, :]), axis=0)
        del flow[var], flowc[var]

    vars = ['U', 'V']
    for var in vars:
        flowm[var] = np.concatenate((flowc[var][:, ic], flow[var][:, i]), axis=1)
        del flow[var], flowc[var]

    import h5py
    path_save = r'F:\AEs_wControl\utils\data\FPc_00k_80k.h5'
    with h5py.File(path_save, 'w') as h5file:
        for key, item in flowm.items():
            h5file.create_dataset(key, data=item)

def save_latent_vector_test():
    import h5py
    import numpy as np
    path_out = r'F:\AEs_wControl\misc\5th_FP_control\CNN-VAE_lr_1e-05_nepoch_500_batch_256_beta_0.005_nr_30_nt_10000_val_2_out.h5'
    path_save = r'F:\AEs_wControl\DATA\FPcz_00k_10k_CNNVAE.npy'

    out = {}
    with h5py.File(path_out, 'r') as f:
        for i in f.keys():
            out[i] = f[i][()]
    np.save(path_save, out['z_test'])

def save_aer_forces():
    import h5py
    import pandas as pd
    import numpy as np

    path_cl = r'F:\Re150\Indipendent_3\Validation\ClValues.txt'
    path_cd = r'F:\Re150\Indipendent_3\Validation\CdValues.txt'
    path_save = r'F:\AEs_wControl\DATA\FPcf_00k_03k.npy'

    data_cl = pd.read_csv(path_cl, header=None, sep='\t')
    data_cd = pd.read_csv(path_cd, header=None, sep='\t')

    forces = {'CL': {'F': data_cl[1].to_numpy(), 'T': data_cl[2].to_numpy(), 'B': data_cl[3].to_numpy(),
                     'tot': data_cl[1].to_numpy() + data_cl[2].to_numpy() + data_cl[2].to_numpy()},
              'CD': {'F': data_cd[1].to_numpy(), 'T': data_cd[2].to_numpy(), 'B': data_cd[3].to_numpy(),
                     'tot': data_cd[1].to_numpy() + data_cd[2].to_numpy() + data_cd[2].to_numpy()}}

    np.save(path_save, forces)

def get_control_seq():
    import h5py
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy.io as sio

    path = r'F:\Re150\InputValues'
    inputs = 3

    for i in range(1, inputs + 1):
        df = pd.read_csv(path + str(i) + '.txt', header=None, sep=' ')
        if i == 1:
            utrain = df.to_numpy()
        else:
            utrain = np.concatenate((utrain, df.to_numpy()), axis=0)

    df = pd.read_csv(path + str(4) + '.txt', header=None, sep=' ')
    uval = df.to_numpy()

    nt_train = np.shape(utrain)[0]
    nt_val = np.shape(uval)[0]

    # 0: F, 1: T, 2: B
    BB = []
    BT = []
    M = []
    M_val = []
    FSP = []

    for t in range(nt_val):
        if (uval[t, 1] == uval[t, 2]) & (uval[t, 1] == uval[t, 0]):  # magnus
            M_val.append(t)

    for t in range(nt_train):
        if (utrain[t, 1] == -utrain[t, 2]) & (utrain[t, 1] > 0) & (utrain[t, 0] == 0):  # base bleed, T-CCW, B-CW
            BB.append(t)
        if (utrain[t, 1] == -utrain[t, 2]) & (utrain[t, 1] < 0) & (utrain[t, 0] == 0):  # boat tailing, T-CW, B-CCW
            BT.append(t)
        if (utrain[t, 1] == utrain[t, 2]) & (utrain[t, 1] == utrain[t, 0]):  # magnus
            M.append(t)
        if (utrain[t, 1] == 0) & (utrain[t, 2] == 0) & (utrain[t, 0] != 0):  # forward stagnation point
            FSP.append(t)

    BB = np.array(BB)
    BT = np.array(BT)
    M = np.array(M)
    M_val = np.array(M_val)
    FSP = np.array(FSP)

    jBB = np.where((BB[1:] - BB[:-1]) > 1)
    jBT = np.where((BT[1:] - BT[:-1]) > 1)
    jM = np.where((M[1:] - M[:-1]) > 1)
    jFSP = np.where((FSP[1:] - FSP[:-1]) > 1)

    a = 0
    # iBB1 = np.arange(30174, 30401)
    # iBB2 = np.arange(31274, 31576)

    # iBT1 = np.arange(37874, 38176)
    # iBT2 = np.arange(39049, 39276)

    # iFSP1 = np.arange(7075, 7377)
    # iFSP2 = np.arange(20825, 21127)
    # iFSP3 = np.arange(48324, 48626)
    # iFSP4 = np.arange(62074, 62376)

    # iM1 = np.arange(0, 777)
    # iM2 = np.arange(23024, 23326)
    # iM3 = np.arange(34574, 34876)
    # iM4 = np.arange(46124, 46426)
    # iM5 = np.arange(68674, 69452)

    # iMval = np.arange(0,293)

    indices = np.arange(31274, 31576)
    Mdict = {'vF': utrain[indices, 0], 'vT': utrain[indices, 1], 'vB': utrain[indices, 2]}
    rootout = r'\FPc_BB_2_u.mat'
    sio.savemat(r'F:\AEs_wControl\utils\data' + rootout, Mdict)

def get_control_lift_drag():
    import h5py
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy.io as sio

    path_cl = r'F:\Re150\ClValues'
    path_cd = r'F:\Re150\CdValues'
    inputs = 3

    for i in range(1, inputs + 1):
        df = pd.read_csv(path_cl + str(i) + '.txt', header=None, sep='\t')
        if i == 1:
            cltrain = df.to_numpy()[:, 1:]
        else:
            cltrain = np.concatenate((cltrain, df.to_numpy()[:, 1:]), axis=0)
    cltrain = np.concatenate((cltrain, (cltrain[:, 0] + cltrain[:, 1] + cltrain[:, 2]).reshape(-1, 1)), axis=1)

    df = pd.read_csv(path_cl + str(4) + '.txt', header=None, sep='\t')
    clval = df.to_numpy()[:, 1:]

    for i in range(1, inputs + 1):
        df = pd.read_csv(path_cd + str(i) + '.txt', header=None, sep='\t')
        if i == 1:
            cdtrain = df.to_numpy()[:, 1:]
        else:
            cdtrain = np.concatenate((cdtrain, df.to_numpy()[:, 1:]), axis=0)
    cdtrain = np.concatenate((cdtrain, (cdtrain[:, 0] + cdtrain[:, 1] + cdtrain[:, 2]).reshape(-1, 1)), axis=1)

    df = pd.read_csv(path_cl + str(4) + '.txt', header=None, sep='\t')
    cdval = df.to_numpy()[:, 1:]

    nt_train = np.shape(cltrain)[0]
    nt_val = np.shape(clval)[0]
    t_train = np.arange(0, nt_train) * 0.1
    t_val = np.arange(0, nt_val) * 0.1

    iBB1 = np.arange(30174, 30401)
    iBB2 = np.arange(31274, 31576)

    iBT1 = np.arange(37874, 38176)
    iBT2 = np.arange(39049, 39276)

    iFSP1 = np.arange(7075, 7377)
    iFSP2 = np.arange(20825, 21127)
    iFSP3 = np.arange(48324, 48626)
    iFSP4 = np.arange(62074, 62376)

    iM1 = np.arange(0, 777)
    iM2 = np.arange(23024, 23326)
    iM3 = np.arange(34574, 34876)
    iM4 = np.arange(46124, 46426)
    iM5 = np.arange(68674, 69452)

    iMval = np.arange(0, 293)

    i = 0
    fig, ax = plt.subplots(1, 1)
    ax.plot(t_train, cltrain[:, i], '-', color='gray')
    ax.plot(t_train[iBB1], cltrain[iBB1, i], '-', color='b', label='BB')
    ax.plot(t_train[iBB2], cltrain[iBB2, i], '-', color='b', label='_nolegend_')
    ax.plot(t_train[iBT1], cltrain[iBT1, i], '-', color='m', label='BT')
    ax.plot(t_train[iBT2], cltrain[iBT2, i], '-', color='m', label='_nolegend_')
    ax.plot(t_train[iFSP1], cltrain[iFSP1, i], '-', color='r', label='FSP')
    ax.plot(t_train[iFSP2], cltrain[iFSP2, i], '-', color='r', label='_nolegend_')
    ax.plot(t_train[iFSP3], cltrain[iFSP3, i], '-', color='r', label='_nolegend_')
    ax.plot(t_train[iFSP4], cltrain[iFSP4, i], '-', color='r', label='_nolegend_')
    ax.plot(t_train[iM1], cltrain[iM1, i], '-', color='g', label='M')
    ax.plot(t_train[iM2], cltrain[iM2, i], '-', color='g', label='_nolegend_')
    ax.plot(t_train[iM3], cltrain[iM3, i], '-', color='g', label='_nolegend_')
    ax.plot(t_train[iM4], cltrain[iM4, i], '-', color='g', label='_nolegend_')
    ax.plot(t_train[iM5], cltrain[iM5, i], '-', color='g', label='_nolegend_')

    ax.legend()
    ax.grid()
    ax.set_ylabel('$C_{D_F}$')
    ax.set_xlabel(r'$\tau$')

def prepare_latent():
    import h5py
    import numpy as np

    from utils.data.read_data import read_FP

    path_grid = r'F:\AEs_wControl\DATA\FP_grid.h5'
    path_flow = r'F:\AEs_wControl\DATA\FP_14k_24k.h5'
    path_out = r'F:\AEs_wControl\DATA\latent\FPz_14k_24k_CCNNAE.h5'

    grid, flow = read_FP(path_grid, path_flow)

    inn = {}
    with h5py.File(path_out, 'r') as f:
        for i in f.keys():
            inn[i] = f[i][()]

    out = {}
    out['Z'] = inn['z_test']
    out['t'] = flow['t']

    with h5py.File(path_out, 'w') as h5file:
        for key, item in out.items():
            h5file.create_dataset(key, data=item)