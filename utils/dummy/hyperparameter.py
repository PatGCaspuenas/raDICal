def refinements():
    # PACKAGES
    import os

    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.io as sio
    import numpy as np
    import re
    import random
    from matplotlib.ticker import FuncFormatter

    from utils.data.read_data import read_FP
    from utils.POD.fits import elbow_fit

    # PARAMETERS
    rtol = 1e-3
    patience = 10

    path_out = r'F:\AEs_wControl\OUTPUT2\2nd_control'
    path_list = r'F:\AEs_wControl\OUTPUT2\2nd_FP_control.csv'
    path_data = r'F:\AEs_wControl\DATA\FPc_00k_70k.h5'
    path_grid = r'F:\AEs_wControl\DATA\FP_grid.h5'
    nt_POD = 1000

    files = os.listdir(path_out)
    files = [f for f in files if f.endswith('.mat')]

    df = pd.read_csv(path_list)
    df['nepoch_s'] = 0
    df['l_train_s'] = 0
    df['l_val_s'] = 0
    df['l_train_f'] = 0
    df['l_val_f'] = 0
    df['RMSE_AE'] = 0
    df['CEA'] = 0
    df['Sc'] = 0
    df['zdetR'] = 0
    df['Dtime'] = 0

    for f in files:
        # Get hyperparameters of file
        vars = re.split('_|c|lr|nepoch|batch|beta|nr|nt|history.mat', f)
        vars = list(filter(None, vars))
        AE, lr, n_epochs, batch_size, beta, nr, nt = vars
        lr, n_epochs, batch_size, beta, nr, nt = float(lr), int(float(n_epochs)), int(float(batch_size)), float(
            beta), int(float(nr)), int(float(nt))

        # Match with csv list
        ilist = df.loc[(df['lr'] == lr) & (df['n_epochs'] == n_epochs) & (df['batch_size'] == batch_size)
                       & (df['beta'] == beta) & (df['nr'] == nr) & (df['nt'] == nt) & (df['nr'] == nr) & (
                                   df['AE'] == AE)].index[0]

        M = sio.loadmat(os.path.join(path_out, f))

        rerr = np.abs(M['val_energy_loss'][0, 1:] - M['val_energy_loss'][0, :-1]) / M['val_energy_loss'][0, :-1]
        irerr = np.where(rerr < rtol)[0]
        i_s = irerr[patience] if (len(irerr) > patience) else (len(rerr))

        df['nepoch_s'][ilist] = i_s + 1
        df['l_train_s'][ilist] = M['energy_loss'][0, i_s]
        df['l_val_s'][ilist] = M['val_energy_loss'][0, i_s]
        df['l_train_f'][ilist] = M['energy_loss'][0, -1]
        df['l_val_f'][ilist] = M['val_energy_loss'][0, -1]
        df['CEA'][ilist] = M['CEA'][0][0]
        df['Sc'][ilist] = M['Sc'][0][0]
        df['RMSE_AE'][ilist] = M['RMSE_AE'][0][0]
        df['zdetR'][ilist] = M['zdetR'][0][0]
        df['Dtime'][ilist] = M['Dtime'][0][0] / 60

    # SC AGAINST DETR
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.semilogy(df.loc[df['AE'] == 'CNN-VAE']['nr'].to_numpy(), df.loc[df['AE'] == 'CNN-VAE']['Sc'].to_numpy(), 'o-',
                 color='k', label='CNN-VAE')
    ax1.semilogy(df.loc[df['AE'] == 'C-CNN-AE']['nr'].to_numpy(), df.loc[df['AE'] == 'C-CNN-AE']['Sc'].to_numpy(), '*-',
                 color='k', label='C-CNN-AE')
    ax1.semilogy(df.loc[df['AE'] == 'CNN-VAE']['nr'].to_numpy(), df.loc[df['AE'] == 'CNN-VAE']['Sc'].to_numpy(), 'o-',
                 color='g', label='_nolegend_')
    ax1.semilogy(df.loc[df['AE'] == 'C-CNN-AE']['nr'].to_numpy(), df.loc[df['AE'] == 'C-CNN-AE']['Sc'].to_numpy(), '*-',
                 color='g', label='_nolegend_')

    ax2.semilogy(df.loc[df['AE'] == 'CNN-VAE']['nr'].to_numpy(), df.loc[df['AE'] == 'CNN-VAE']['zdetR'].to_numpy(),
                 'o-', color='m', label='_nolegend_')
    ax2.semilogy(df.loc[df['AE'] == 'C-CNN-AE']['nr'].to_numpy(), df.loc[df['AE'] == 'C-CNN-AE']['zdetR'].to_numpy(),
                 '*-', color='m', label='_nolegend_')

    ax1.set_xlabel('$r$')
    ax1.set_ylabel('$S_c$', color='g')
    ax2.set_ylabel('$det_R$', color='m')

    for axis in [ax1.yaxis]:
        formatter = FuncFormatter(lambda y, _: '{:.2g}'.format(y))
        axis.set_minor_formatter(formatter)
        axis.set_major_formatter(formatter)

    ax1.legend()
    ax1.grid()

    plt.show()

    # GET POD
    grid, flow = read_FP(path_grid, path_data)
    D = np.concatenate((flow['U'], flow['V']), axis=0)
    Dmean = np.mean(D, axis=1).reshape((np.shape(D)[0]), 1)
    Ddt = D - Dmean
    del D, flow

    nt = np.shape(Ddt)[1]
    i_snps = random.sample([*range(nt)], nt_POD)

    Phi, Sigma, Psi = np.linalg.svd(Ddt[:, i_snps], full_matrices=False)
    CE = np.cumsum(Sigma ** 2) / np.sum(Sigma ** 2)

    # PLOT
    nr = 100
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(box_aspect=1))

    ax.loglog(np.arange(1, nt_POD + 1), CE, 'bo-', label='POD')
    ax.loglog(df.loc[df['AE'] == 'CNN-VAE']['nr'].to_numpy(), df.loc[df['AE'] == 'CNN-VAE']['CEA'].to_numpy(), 'mo-',
              label='CNN-VAE')
    ax.loglog(df.loc[df['AE'] == 'C-CNN-AE']['nr'].to_numpy(), df.loc[df['AE'] == 'C-CNN-AE']['CEA'].to_numpy(), 'go-',
              label='C-CNN-AE')

    ax.set_xlabel('$r$')
    ax.set_ylabel('CEA')

    ax.axis([1, nr, 0.1, 1])
    for axis in [ax.yaxis]:
        formatter = FuncFormatter(lambda y, _: '{:.2g}'.format(y))
        axis.set_minor_formatter(formatter)
        axis.set_major_formatter(formatter)
    ax.legend()
    plt.show()

def last_refinement():
    # PACKAGES
    import h5py
    import numpy as np
    import warnings
    import matplotlib.pyplot as plt
    import random
    import os
    import pandas as pd

    # LOCAL FILES
    from utils.data.read_data import read_FP
    from utils.plt.plt_control import plot_input
    from utils.plt.plt_snps import *
    from utils.plt.plt_config import *
    from utils.plt.plt_AE import *
    from utils.modelling.errors_flow import get_latent_correlation_matrix
    from utils.AEs.outputs import filter_AE_z

    path_flow = r'F:\AEs_wControl\OUTPUT2\3rd_FP_control_smooth\CNN-VAE_lr_1e-5_nepoch_500_batch_256_beta_5e-3_nr_15_nt_10000_SG3_15_out.h5'
    path_true = r'F:\AEs_wControl\DATA\FPc_00k_03k.h5'
    path_grid = r'F:\AEs_wControl\DATA\FP_grid.h5'

    # LOAD DATA
    true = {}
    with h5py.File(path_true, 'r') as f:
        for i in f.keys():
            true[i] = f[i][()]
    D = np.concatenate((true['U'], true['V']), axis=0)
    Dmean = np.mean(D, axis=1).reshape((np.shape(D)[0]), 1)
    Ddt = D - Dmean
    t = true['t']
    u = np.concatenate((true['vF'], true['vT'], true['vB']), axis=1)

    flow = {}
    with h5py.File(path_flow, 'r') as f:
        for i in f.keys():
            flow[i] = f[i][()]

    grid = {}
    with h5py.File(path_grid, 'r') as f:
        for i in f.keys():
            grid[i] = f[i][()]

    i_order = np.arange(0, 500, 5)
    nr = np.shape(flow['z_test'])[1]

    flow['z_test'] = filter_AE_z(flow['z_test'])

    import matplotlib as mpl
    cmap = mpl.cm.get_cmap('jet')
    clist = cmap(np.linspace(0, 1, nr))

    nrows = int(np.ceil(nr / 3))
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, layout='tight')

    c = 0
    for nrow in range(nrows):
        for ncol in range(ncols):
            ax[nrow, ncol].plot(t[:, 0], flow['z_test'][:, c], 'k-', linewidth=2)
            ax[nrow, ncol].text(1, -0.75, '$i=' + str(c + 1) + '$', color='r', fontsize=8)
            c += 1
            if nrow == nrows - 1:
                ax[nrow, ncol].set_xlabel(r'$t/\tau$')
            else:
                ax[nrow, ncol].set_xticks([])

            if ncol == 0:
                ax[nrow, ncol].set_ylabel('$z_i$')
            else:
                ax[nrow, ncol].set_yticks([])

            ax[nrow, ncol].set_xlim([0, 50])
            ax[nrow, ncol].set_ylim([-0.75, 0.75])

    plt.show()

    zdetR, zRij = get_latent_correlation_matrix(flow['z_test'])
    plot_corr_matrix(zRij, zdetR)

    plot_video_snp(grid, Ddt[:, i_order], 'anim.gif', limits=[-0.5, 0.5], make_axis_visible=[1, 1], show_title=0,
                   show_colorbar=0, flag_flow='FP')
    plot_video_snp(grid, flow['Dr_test_AE'][:, i_order], 'anim2.gif', limits=[-0.5, 0.5], make_axis_visible=[1, 1],
                   show_title=0, show_colorbar=0, flag_flow='FP')
    plot_video_snp(grid, np.abs(flow['Dr_test_AE'][:, i_order] - Ddt[:, i_order]), 'anim3.gif', limits=[0, 0.2],
                   make_axis_visible=[1, 1], show_title=0, show_colorbar=0, flag_flow='FP')

    plot_video_snp(grid, Ddt[:, i_order], 'anim.gif', limits=[-0.5, 0.5], make_axis_visible=[1, 1], show_title=0,
                   show_colorbar=0, flag_flow='FP', flag_control=1, u=u[i_order, :], t=t[i_order, :])
    plot_video_snp(grid, flow['Dr_test_AE'][:, i_order], 'anim2.gif', limits=[-0.5, 0.5], make_axis_visible=[1, 1],
                   show_title=0, show_colorbar=0, flag_flow='FP', flag_control=1, u=u[i_order, :], t=t[i_order, :])
    plot_video_snp(grid, np.abs(flow['Dr_test_AE'][:, i_order] - Ddt[:, i_order]), 'anim3.gif', limits=[0, 0.2],
                   make_axis_visible=[1, 1], show_title=0, show_colorbar=0, flag_flow='FP', flag_control=1,
                   u=u[i_order, :], t=t[i_order, :])

def refinement_beta_grid():
    # PACKAGES
    import os

    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.io as sio
    import numpy as np
    import re
    import random
    from matplotlib.ticker import FuncFormatter

    from utils.data.read_data import read_FP
    from utils.POD.fits import elbow_fit

    # PARAMETERS
    rtol = 1e-3
    patience = 10

    path_out = r'F:\AEs_wControl\misc\4th_FP_no_control'
    path_list = r'F:\AEs_wControl\OUTPUT\4th_FP_no_control.csv'
    path_data = r'F:\AEs_wControl\DATA\FPc_00k_70k.h5'
    path_grid = r'F:\AEs_wControl\DATA\FP_grid.h5'

    files = os.listdir(path_out)
    files = [f for f in files if f.endswith('.mat')]

    df = pd.read_csv(path_list)
    df['nepoch_s'] = 0
    df['l_train_s'] = 0
    df['l_val_s'] = 0
    df['l_train_f'] = 0
    df['l_val_f'] = 0
    df['RMSE_AE'] = 0
    df['CEA'] = 0
    df['Sc'] = 0
    df['zdetR'] = 0
    df['Dtime'] = 0

    for f in files:
        # Get hyperparameters of file
        vars = re.split('_|c|lr|nepoch|batch|beta|nr|nt|history.mat', f)
        vars = list(filter(None, vars))
        AE, lr, n_epochs, batch_size, beta, nr, nt = vars
        lr, n_epochs, batch_size, beta, nr, nt = float(lr), int(float(n_epochs)), int(float(batch_size)), float(
            beta), int(float(nr)), int(float(nt))

        # Match with csv list
        ilist = df.loc[(df['lr'] == lr) & (df['n_epochs'] == n_epochs) & (df['batch_size'] == batch_size)
                       & (df['beta'] == beta) & (df['nr'] == nr) & (df['nt'] == nt) & (df['nr'] == nr) & (
                               df['AE'] == AE)].index[0]

        M = sio.loadmat(os.path.join(path_out, f))

        rerr = np.abs(M['val_energy_loss'][0, 1:] - M['val_energy_loss'][0, :-1]) / M['val_energy_loss'][0, :-1]
        irerr = np.where(rerr < rtol)[0]
        i_s = irerr[patience] if (len(irerr) > patience) else (len(rerr))

        df['nepoch_s'][ilist] = i_s + 1
        df['l_train_s'][ilist] = M['energy_loss'][0, i_s]
        df['l_val_s'][ilist] = M['val_energy_loss'][0, i_s]
        df['l_train_f'][ilist] = M['energy_loss'][0, -1]
        df['l_val_f'][ilist] = M['val_energy_loss'][0, -1]
        df['CEA'][ilist] = M['CEA'][0][0]
        df['Sc'][ilist] = M['Sc'][0][0]
        df['RMSE_AE'][ilist] = M['RMSE_AE'][0][0]
        df['zdetR'][ilist] = M['zdetR'][0][0]
        df['Dtime'][ilist] = M['Dtime'][0][0] / 60

    # CREATE GRID

    nrs = 9
    betas = 5

    BETA, NR, CEA, SC, ZDETR, LVALF = np.zeros((6, betas, nrs))
    i_beta_control = [3, 2, 1, 0, 4]
    i_beta_no_control = [3, 0, 1, 2, 4]

    for i in range(betas):
        index = i_beta_no_control[i]
        BETA[index, :] = df['beta'][i * nrs:(i + 1) * nrs].to_numpy()
        NR[index, :] = df['nr'][i * nrs:(i + 1) * nrs].to_numpy()
        CEA[index, :] = df['CEA'][i * nrs:(i + 1) * nrs].to_numpy()
        SC[index, :] = df['Sc'][i * nrs:(i + 1) * nrs].to_numpy()
        ZDETR[index, :] = df['zdetR'][i * nrs:(i + 1) * nrs].to_numpy()
        LVALF[index, :] = df['l_val_f'][i * nrs:(i + 1) * nrs].to_numpy()

    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots(1, 1)
    surf = ax.contourf(BETA, NR, SC, cmap='coolwarm',
                       linewidth=0, antialiased=False)
    ax.set_ylabel('$n_r$')
    ax.set_xlabel(r'$\beta$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('$S_c$')
    fig.colorbar(surf)
    for axis in [ax.yaxis]:
        formatter = FuncFormatter(lambda y, _: '{:g}'.format(y))
        axis.set_minor_formatter(formatter)
        axis.set_major_formatter(formatter)

    fig, ax = plt.subplots(1, 1)
    surf = ax.contourf(BETA, NR, CEA, cmap='coolwarm',
                       linewidth=0, antialiased=False)
    ax.set_ylabel('$n_r$')
    ax.set_xlabel(r'$\beta$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('$CEA$')
    fig.colorbar(surf)
    for axis in [ax.yaxis]:
        formatter = FuncFormatter(lambda y, _: '{:g}'.format(y))
        axis.set_minor_formatter(formatter)
        axis.set_major_formatter(formatter)

    fig, ax = plt.subplots(1, 1)
    surf = ax.contourf(BETA, NR, np.log10(LVALF), cmap='coolwarm',
                       linewidth=0, antialiased=False)
    ax.set_ylabel('$n_r$')
    ax.set_xlabel(r'$\beta$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('$log10(L_{val})$')
    fig.colorbar(surf)
    for axis in [ax.yaxis]:
        formatter = FuncFormatter(lambda y, _: '{:g}'.format(y))
        axis.set_minor_formatter(formatter)
        axis.set_major_formatter(formatter)

    plt.show()

def refinements_new():
    # PACKAGES
    import os

    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.io as sio
    import numpy as np
    import re
    import random
    from matplotlib.ticker import FuncFormatter

    from utils.data.read_data import read_FP
    from utils.POD.fits import elbow_fit

    # PARAMETERS

    path_out = r'F:\AEs_wControl\misc\1st_control'
    path_list = r'F:\AEs_wControl\OUTPUT\1st_FP.csv'

    files = os.listdir(path_out)
    files = [f for f in files if f.endswith('.mat')]

    df = pd.read_csv(path_list)
    df['nepoch_s'] = 0
    df['l_train_s'] = 0
    df['l_val_s'] = 0
    df['l_train_f'] = 0
    df['l_val_f'] = 0
    df['RMSE_AE'] = 0
    df['CEA'] = 0
    df['Sc'] = 0
    df['zdetR'] = 0
    df['zmeanR'] = 0
    df['Dtime'] = 0

    for f in files:
        # Get hyperparameters of file
        vars = re.split('_|c|lr|nepoch|batch|beta|nr|nt|history.mat', f)
        vars = list(filter(None, vars))
        AE, lr, n_epochs, batch_size, beta, nr, nt = vars
        lr, n_epochs, batch_size, beta, nr, nt = float(lr), int(float(n_epochs)), int(float(batch_size)), float(
            beta), int(float(nr)), int(float(nt))

        # Match with csv list
        ilist = df.loc[(df['lr'] == lr) & (df['n_epochs'] == n_epochs) & (df['batch_size'] == batch_size)
                       & (df['beta'] == beta) & (df['nr'] == nr) & (df['nt'] == nt) & (df['nr'] == nr) & (
                               df['AE'] == AE)].index[0]

        M = sio.loadmat(os.path.join(path_out, f))

        i_s = len(M['val_energy_loss'][0, :]) - 1

        df['nepoch_s'][ilist] = i_s + 1
        df['l_train_s'][ilist] = M['energy_loss'][0, i_s]
        df['l_val_s'][ilist] = M['val_energy_loss'][0, i_s]
        df['l_train_f'][ilist] = M['energy_loss'][0, -1]
        df['l_val_f'][ilist] = M['val_energy_loss'][0, -1]
        df['CEA'][ilist] = M['CEA'][0][0]
        df['Sc'][ilist] = M['Sc'][0][0]
        df['RMSE_AE'][ilist] = M['RMSE_AE'][0][0]
        df['zmeanR'][ilist] = M['zmeanR'][0][0]
        df['Dtime'][ilist] = M['Dtime'][0][0] / 60
        df['zdetR'][ilist] = M['zdetR'][0][0]

def last_refinement_new():
    # PACKAGES
    import os
    import h5py

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import random

    from utils.data.read_data import read_FP
    from utils.modelling.errors_z import get_frequencies, get_latent_correlation_matrix
    from utils.POD.fits import elbow_fit
    from utils.plt.plt_AE import plt_PSD_lat, plot_Lissajous, plot_corr_matrix
    from utils.plt.plt_snps import plot_video_snp
    from utils.plt.plt_POD import plt_psi

    # PARAMETERS

    path_out = r'F:\AEs_wControl\misc\3rd_control'
    path_list = r'F:\AEs_wControl\OUTPUT\3rd_FP_control.csv'
    path_data = r'F:\AEs_wControl\DATA\FPc_00k_70k.h5'
    path_grid = r'F:\AEs_wControl\DATA\FP_grid.h5'
    nt_POD = 1000

    files = os.listdir(path_out)
    files = [f for f in files if f.endswith('.h5')]

    df = pd.read_csv(path_list)

    for file in files:

        if 'CNN-VAE' in file:
            CNNVAE = {}
            with h5py.File(os.path.join(path_out, file), 'r') as f:
                for i in f.keys():
                    CNNVAE[i] = f[i][()]
        else:
            CCNNAE = {}
            with h5py.File(os.path.join(path_out, file), 'r') as f:
                for i in f.keys():
                    CCNNAE[i] = f[i][()]

    # GET POD
    grid, flow = read_FP(path_grid, path_data)
    D = np.concatenate((flow['U'], flow['V']), axis=0)
    Dmean = np.mean(D, axis=1).reshape((np.shape(D)[0]), 1)
    Ddt = D - Dmean
    del D, flow

    nt = np.shape(Ddt)[1]
    i_snps = random.sample([*range(nt)], nt_POD)

    Phi, Sigma, Psi = np.linalg.svd(Ddt[:, i_snps], full_matrices=False)
    CE = np.cumsum(Sigma ** 2) / np.sum(Sigma ** 2)

    del Ddt
    path_data = r'F:\AEs_wControl\DATA\FPc_00k_03k.h5'
    grid, flow = read_FP(path_grid, path_data)
    D = np.concatenate((flow['U'], flow['V']), axis=0)
    Ddt = D - Dmean
    t = flow['t']
    # u = np.concatenate((flow['vF'], flow['vT'], flow['vB']), axis=1)
    del D, flow

    Psi_new = np.dot(Phi.T, Ddt) / np.linalg.norm(np.dot(Phi.T, Ddt), axis=1).reshape(-1, 1)
    Psi_new = np.dot(Phi.T, Ddt) / np.max(np.abs(np.dot(Phi.T, Ddt)), axis=1).reshape(-1, 1)

    plt_PSD_lat(Psi_new[:, 0:30], 10)
    # CNN-VAE c : 90.1
    # C-CNN-AE c: 95.8
    a = 0

    detR, meanR, Rij = get_latent_correlation_matrix(CCNNAE['z_test'])
    plot_corr_matrix(Rij, detR)

    it = np.arange(0, 500, 5)
    plot_video_snp(grid, CNNVAE['Dr_test_AE'][:, it], 'test.gif', limits=[-0.5, 0.5], make_axis_visible=[1, 1],
                   show_title=0, show_colorbar=0, flag_flow='FP', flag_control=0)
    plot_video_snp(grid, np.abs(CNNVAE['Dr_test_AE'][:, it] - Ddt[:, it]), 'test2.gif', limits=[0, 0.2],
                   make_axis_visible=[1, 1], show_title=0, show_colorbar=0, flag_flow='FP', flag_control=0)

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(box_aspect=1))

    nr = np.shape(CNNVAE['z_test'])[1]
    freq = get_frequencies(CNNVAE['z_test'])
    for i in range(nr):
        for j in range(len(freq[i])):
            ax.semilogy(i, 1 / freq[i][j], 'ko', markersize=7)

    nr = np.shape(CCNNAE['z_test'])[1]
    freq = get_frequencies(CCNNAE['z_test'])
    for i in range(nr):
        for j in range(len(freq[i])):
            ax.semilogy(i, 1 / freq[i][j], 'ro', markersize=4)

    nr = np.shape(Psi_new)[0]
    freq = get_frequencies(Psi_new[0:32, :].T)
    for i in range(nr):
        for j in range(len(freq[i])):
            ax.semilogy(i, 1 / freq[i][j], 'bo', markersize=2)

    ax.set_xlabel('$n_r$')
    ax.set_ylabel('$T$')
