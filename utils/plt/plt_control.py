from utils.modelling.control import rotation2coords

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_rotation(ax, it, x, y, flag_flow):

    if flag_flow == 'FP':

        xF, yF = -3 / 2 * np.cos(30 * np.pi / 180), 0
        xB, yB = 0, -3 / 4
        xT, yT = 0, 3 / 4

        ax.plot([xF, x['F'][it]], [yF, y['F'][it]], linewidth=1.3, color='white')
        ax.plot([xT, x['T'][it]], [yT, y['T'][it]], linewidth=1.3, color='white')
        ax.plot([xB, x['B'][it]], [yB, y['B'][it]], linewidth=1.3, color='white')


def generate_rotation(flag_flow, u, t):

    nt = len(t)

    if flag_flow == 'FP':

        R = 0.5
        xF, yF = -3 / 2 * np.cos(30 * np.pi / 180), 0
        xB, yB = 0, -3 / 4
        xT, yT = 0, 3 / 4

        x, y = {}, {}

        x['F'], y['F'] = np.zeros((2, nt))
        x['T'], y['T'] = np.zeros((2, nt))
        x['B'], y['B'] = np.zeros((2, nt))

        x['F'][0], y['F'][0] = 0.5 + xF, 0 + yF
        x['T'][0], y['T'][0] = 0.5 + xT, 0 + yT
        x['B'][0], y['B'][0] = 0.5 + xB, 0 + yB

        for it in range(1,nt):
            x['F'][it], y['F'][it] = rotation2coords(x['F'][it - 1], y['F'][it - 1], xF, yF, u[it - 1, 0], t[it] - t[it-1], R=R)
            x['T'][it], y['T'][it] = rotation2coords(x['T'][it - 1], y['T'][it - 1], xT, yT, u[it - 1, 1], t[it] - t[it - 1], R=R)
            x['B'][it], y['B'][it] = rotation2coords(x['B'][it - 1], y['B'][it - 1], xB, yB, u[it - 1, 2], t[it] - t[it - 1], R=R)

    return x, y

def plot_input(flag_flow,u,t):

    nt = len(t)

    fig, ax = plt.subplots(1,1)
    plt.tight_layout()

    if flag_flow == 'FP':

        def animate(it):
            ax.cla()

            ax.plot(t[0:it+1],u[0:it+1, 0], linewidth=2, color='blue', label='$\omega_F \cdot R$')
            ax.plot(t[0:it + 1], u[0:it+1, 1], linewidth=2, color='magenta', label='$\omega_T \cdot R$')
            ax.plot(t[0:it + 1], u[0:it+1, 2], linewidth=2, color='green', label='$\omega_B \cdot R$')

            ax.set_xlabel(r"$\tau$")
            ax.set_ylabel("$u_i$")

            ax.set_xlim([t[0], t[-1]])
            ax.set_ylim([-1.1, 1.1])

            ax.set_yticks([-1,0,1])
            ax.legend(loc='upper right')
            plt.tight_layout()

        anim = animation.FuncAnimation(fig, animate, frames=nt, interval=200)
        writergif = animation.PillowWriter(fps=2)
        anim.save(r'F:\AEs_wControl\control.gif', writer=writergif)
