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
def PSD_forces():
    # REACTION TIME STUDY
    from scipy import signal
    import matplotlib.pyplot as plt
    import h5py

    path_forces = r'F:\AEs_wControl\DATA\forces\FPcf_00k_70k.h5'
    forces = {}
    with h5py.File(path_forces, 'r') as f:
        for i in f.keys():
            forces[i] = f[i][()]

    fs = 10
    f, PSD = signal.periodogram(forces['Cd_total'][:, 0], fs)

    fig, ax = plt.subplots(1, 1)
    ax.loglog(f, PSD, 'b-')
    ax.set_ylabel('$PSD_{C_l}$')
    ax.set_xlabel('$St_D$')
    ax.grid()
    plt.show()

def reaction_corr():
    # REACTION TIME STUDY
    import matplotlib.pyplot as plt
    from scipy import signal
    import h5py
    import random
    import numpy as np

    fs = 10
    path_lat = r'F:\AEs_wControl\DATA\latent\FPcz_00k_70k_CNNVAE.h5'
    lat = {}
    with h5py.File(path_lat, 'r') as f:
        for i in f.keys():
            lat[i] = f[i][()]

    nrows = 5
    ncols = 6
    fig, ax = plt.subplots(nrows, ncols, subplot_kw=dict(box_aspect=1))
    i, j = 0, 0
    for c, ax in enumerate(fig.axes):

        if j == ncols:
            i += 1
            j = 0

        ax.plot([80, 80], [-1.25, 1.25], 'b-', linewidth=0.7)
        ax.plot([100, 100], [-1.25, 1.25], 'b-', linewidth=0.7)
        ax.plot([lat['t'][0, 0], lat['t'][-1, 0]], [0, 0], '-', color='gray', linewidth=0.8,
                label='_nolegend_')
        ax.plot(lat['t'][:, 0], lat['Z'][:, c], 'k-', linewidth=1.3, label='ground truth')

        ax.text(40, -0.95, '$z_{' + str(c) + '}$', color='r', fontsize=12)

        ax.set_ylim([-1.25, 1.25])
        # ax.set_xlim([lat['t'][0,0], lat['t'][-1,0]])
        ax.set_xlim([40, 140])

        if i == (nrows - 1):
            ax.set_xlabel('$t$ [s]')
            # ax.set_xticks([lat['t'][0,0], lat['t'][-1,0]])
            ax.set_xticks([40, 140])
        else:
            ax.set_xticks([])

        if j == 0:
            ax.set_ylabel('$z$')
            ax.set_yticks([-1, 0, 1])
        else:
            ax.set_yticks([])

        j += 1

    fig, ax = plt.subplots(nrows, ncols, subplot_kw=dict(box_aspect=1))
    i, j = 0, 0
    for c, ax in enumerate(fig.axes):

        if j == ncols:
            i += 1
            j = 0

        f, PSD = signal.periodogram(lat['Z'][:, c], fs)
        ax.loglog(f, PSD, 'b-')
        ax.set_ylim([1e-10, 1e2])

        if i == (nrows - 1):
            ax.set_xlabel('$St$')
        else:
            ax.set_xticks([])

        if j == 0:
            ax.set_ylabel('$PSD$')
        else:
            ax.set_yticks([])

        j += 1

    ii = random.sample(range(69454), 1000)
    CORR = np.sqrt(lat['U'][ii, 0] ** 2 + lat['U'][ii, 1] ** 2 + lat['U'][ii, 2] ** 2)
    CORR = (lat['U'][ii, 2] - lat['U'][ii, 1]) / 2
    CORR = lat['U'][ii, 0] + lat['U'][ii, 1] + lat['U'][ii, 2]
    # CORR = lat['U'][ii,0]
    fig, ax = plt.subplots(nrows, ncols, subplot_kw=dict(box_aspect=1))
    i, j = 0, 0
    for c, ax in enumerate(fig.axes):

        if j == ncols:
            i += 1
            j = 0

        ax.plot(CORR, lat['Z'][ii, c], 'ko', markersize=0.5)

        if i == (nrows - 1):
            ax.set_xlabel('$p_3$')
        else:
            ax.set_xticks([])

        if j == 0:
            ax.set_ylabel('$z$')
            ax.set_yticks([-1, 0, 1])
        else:
            ax.set_yticks([])

        j += 1

    plt.show()