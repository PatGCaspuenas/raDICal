import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import numpy as np

from utils.plt.plt_config import plot_body
from utils.plt.plt_control import plot_rotation, generate_rotation

def plot_mean(grid, D):

    X = grid['X']
    Y = grid['Y']
    B = grid['B']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(D)[1]

    if np.shape(D)[0] == n * m:
        k = 1
    elif np.shape(D)[0] == 2 * n * m:
        k = 2
    else:
        k = 3

    M = np.zeros((m, n, nt, k))
    for i in range(k):
        M[:, :, :, i] = np.reshape(D[( (n * m)*i ):( (n * m)*(i + 1) ), :], (m, n, nt), order='F')

    cticks1 = np.linspace(-0.5, 0.5, 3)
    clevels1 = [-0.5, 0.5]
    cticks2 = np.linspace(0, 1, 3)
    clevels2 = [0, 1]
    str = ['u', 'v', 'w']

    fig, ax = plt.subplots(1, k, layout = 'tight')

    plt.tight_layout()

    for j in range(k):
        if j == 0:
            cp0 = ax[j].pcolormesh(X, Y, M[:, :, :, j].reshape(m,n), cmap = 'jet', vmin = clevels2[0], vmax = clevels2[1])
        else:
            cp0 = ax[j].pcolormesh(X, Y, M[:, :, :, j].reshape(m,n), cmap = 'jet', vmin = clevels1[0], vmax = clevels1[1])
        cp00 = ax[j].contourf(X, Y, B, colors='k')

        ax[j].set_title('$\overline{' + str[j] + '}/U_{\infty}$')
        ax[j].axis('scaled')
        ax[j].set_xlabel('$x/D$')
        ax[j].set_xlim([np.min(X), np.max(X)])
        ax[j].set_ylim([np.min(Y), np.max(Y)])

        divider = make_axes_locatable(ax[j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        if j == 0:
            cbar = fig.colorbar(cp0, ax=ax[j], ticks=cticks2, cax=cax)
        else:
            cbar = fig.colorbar(cp0, ax=ax[j], ticks=cticks1, cax=cax)

        if j == 0:
            ax[j].set_ylabel('$y/D$')
        else:
            ax[j].set_yticks([])

    plt.show()
    plt.tight_layout()

def plot_snp(grid, D, limits = [-0.5, 0.5], make_axis_visible = [1, 1], show_title = 1, show_colorbar = 1, flag_flow = 'SC'):

    X = grid['X']
    Y = grid['Y']
    B = grid['B']

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    if np.shape(D)[0] == n * m:
        k = 1
    elif np.shape(D)[0] == 2 * n * m:
        k = 2
    else:
        k = 3

    M = np.zeros((m, n, k))
    for i in range(k):
        M[:, :, i] = np.reshape(D[( (n * m)*i ):( (n * m)*(i + 1) )], (m, n), order='F')

    cticks = np.linspace(limits[0], limits[1], 3)
    clevels = limits
    str = ['u', 'v', 'w']

    fig, ax = plt.subplots(1, k, layout = 'tight')

    plt.tight_layout()

    if k != 1:
        for j in range(k):

            cp0 = ax[j].pcolormesh(X, Y, M[:, :, j].reshape(m,n), cmap = 'jet', vmin = clevels[0], vmax = clevels[1])
            cp00 = ax[j].contourf(X, Y, B, colors='k')

            if show_title:
                ax[j].set_title('${' + str[j] + '}''/U_{\infty}$')
            ax[j].axis('scaled')
            ax[j].set_xlabel('$x/D$')
            ax[j].set_xlim([np.min(X), np.max(X)])
            ax[j].set_ylim([np.min(Y), np.max(Y)])

            if (j == (k-1)) and show_colorbar:
                divider = make_axes_locatable(ax[j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar =fig.colorbar(cp0, ax=ax[j], ticks=cticks, cax=cax)

            if j == 0:
                ax[j].set_ylabel('$y/D$')
            else:
                ax[j].set_yticks([])

            if not make_axis_visible[0]:
                ax[j].set_xticks([])
                ax[j].set_xlabel('')
            if not make_axis_visible[1]:
                ax[j].set_yticks([])
                ax[j].set_ylabel('')

            plot_body(ax[j], flag_flow)
    else:
        cp0 = ax.pcolormesh(X, Y, M[:, :, 0].reshape(m, n), cmap='jet', vmin=clevels[0], vmax=clevels[1])
        cp00 = ax.contourf(X, Y, B, colors='k')

        if show_title:
            ax.set_title('${' + str[0] + '}''/U_{\infty}$')
        ax.axis('scaled')
        ax.set_xlabel('$x/D$')
        ax.set_ylabel('$y/D$')
        ax.set_xlim([np.min(X), np.max(X)])
        ax.set_ylim([np.min(Y), np.max(Y)])

        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(cp0, ax=ax, ticks=cticks, cax=cax)

        if not make_axis_visible[0]:
            ax.set_xticks([])
            ax.set_xlabel('')
        if not make_axis_visible[1]:
            ax.set_yticks([])
            ax.set_ylabel('')

        plot_body(ax, flag_flow)
    plt.show()
    plt.tight_layout()
def plot_video_snp(grid, D, path_out, limits = [-0.5, 0.5], make_axis_visible = [1, 1], show_title = 1, show_colorbar = 1, flag_flow = 'SC', flag_control = 0, u = [], t = []):

    X = grid['X']
    Y = grid['Y']
    B = grid['B']

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    nt = np.shape(D)[1]

    if np.shape(D)[0] == n * m:
        k = 1
    elif np.shape(D)[0] == 2 * n * m:
        k = 2
    else:
        k = 3

    M = np.zeros((m, n, nt, k))
    for i in range(k):
        M[:, :, :, i] = np.reshape(D[( (n * m)*i ):( (n * m)*(i + 1) ), :], (m, n, nt), order='F')

    if flag_control:
        xc, yc = generate_rotation(flag_flow, u, t)

    cticks = np.linspace(limits[0], limits[1], 3)
    clevels = limits
    str = ['u', 'v', 'w']

    fig, ax = plt.subplots(1, k, layout = 'tight')

    plt.tight_layout()
    if k != 1:
        for j in range(k):
            cp0 = ax[j].pcolormesh(X, Y, M[:, :, 0, j].reshape(m,n), cmap = 'jet', vmin = clevels[0], vmax = clevels[1])
            cp00 = ax[j].contourf(X, Y, B, colors='k')
    else:
        cp0 = ax.pcolormesh(X, Y, M[:, :, 0, 0].reshape(m, n), cmap='jet', vmin=clevels[0], vmax=clevels[1])
        cp00 = ax.contourf(X, Y, B, colors='k')

    def animate(it):
        if k != 1:
            for j in range(k):
                ax[j].cla()

                cp0 = ax[j].pcolormesh(X, Y, M[:, :, it, j].reshape(m, n), cmap='jet', vmin=clevels[0], vmax=clevels[1])
                cp00 = ax[j].contourf(X, Y, B, colors='k')

                if show_title:
                    ax[j].set_title('${' + str[j] + '}''/U_{\infty}$')
                ax[j].axis('scaled')
                ax[j].set_xlabel('$x/D$')
                ax[j].set_xlim([np.min(X), np.max(X)])
                ax[j].set_ylim([np.min(Y), np.max(Y)])

                if (j == (k-1)) and show_colorbar:
                    divider = make_axes_locatable(ax[j])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar =fig.colorbar(cp0, ax=ax[j], ticks=cticks, cax=cax)

                if j == 0:
                    ax[j].set_ylabel('$y/D$')
                else:
                    ax[j].set_yticks([])

                if not make_axis_visible[0]:
                    ax[j].set_xticks([])
                    ax[j].set_xlabel('')
                if not make_axis_visible[1]:
                    ax[j].set_yticks([])
                    ax[j].set_ylabel('')

                plot_body(ax[j], flag_flow)
                if flag_control:
                    plot_rotation(ax[j], it, xc, yc, flag_flow)

        else:
            ax.cla()
            
            cp0 = ax.pcolormesh(X, Y, M[:, :, it, 0].reshape(m, n), cmap='jet', vmin=clevels[0], vmax=clevels[1])
            cp00 = ax.contourf(X, Y, B, colors='k')

            if show_title:
                ax.set_title('${' + str[0] + '}''/U_{\infty}$')
            ax.axis('scaled')
            ax.set_xlabel('$x/D$')
            ax.set_ylabel('$y/D$')
            ax.set_xlim([np.min(X), np.max(X)])
            ax.set_ylim([np.min(Y), np.max(Y)])

            if show_colorbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(cp0, ax=ax, ticks=cticks, cax=cax)

            if not make_axis_visible[0]:
                ax.set_xticks([])
                ax.set_xlabel('')
            if not make_axis_visible[1]:
                ax.set_yticks([])
                ax.set_ylabel('')

            plot_body(ax, flag_flow)
            if flag_control:
                plot_rotation(ax, it, xc, yc, flag_flow)

        plt.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=nt, interval=200)
    writergif = animation.PillowWriter(fps=2)
    anim.save(path_out, writer=writergif)




