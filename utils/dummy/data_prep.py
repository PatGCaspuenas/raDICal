

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