def plot_lic_control_cases():
    import h5py
    import numpy as np
    type_flow = 'M'
    num = 5
    path = r'F:\AEs_wControl\utils\data\FPc_' + type_flow + '_' + str(num) + '.h5'
    path_out = 'M_5.gif'
    path_grid = r'F:\AEs_wControl\DATA\FP_grid.h5'
    path_mean = r'F:\AEs_wControl\utils\data\FPc_Dmean.npy'

    flow = {}
    with h5py.File(path, 'r') as f:
        for i in f.keys():
            flow[i] = f[i][()]

    grid = {}
    with h5py.File(path_grid, 'r') as f:
        for i in f.keys():
            grid[i] = f[i][()]

    Dmean = np.load(path_mean)
    from utils.plt.plt_snps import plot_video_snp_lic

    D = np.concatenate((flow['U'], flow['V']), axis=0)
    D = D - Dmean
    u = np.concatenate((flow['vF'].T, flow['vT'].T, flow['vB'].T), axis=1)
    t = flow['t']
    nt = len(t)

    it = np.arange(0, nt, 5)
    plot_video_snp_lic(grid, D[:, it], path_out, limits=[-2, 2], make_axis_visible=[1, 1], show_title=0,
                       show_colorbar=1,
                       flag_flow='FP', flag_control=1, u=u[it, :], t=t[it, :])