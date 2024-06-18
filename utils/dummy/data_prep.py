

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