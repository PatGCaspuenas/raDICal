import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

path_forces = r'F:\AEs_wControl\DATA\FPc_00k_03k.h5'
forces = {}
with h5py.File(path_forces, 'r') as f:
    for i in f.keys():
        forces[i] = f[i][()]

tmin, tmax = 40,140
t_delay = 0.5

fig, ax = plt.subplots(4,1)
ax[0].plot(forces['t'],forces['vF'],'m',label='F')
ax[0].plot(forces['t'],forces['vT'],'b',label='T')
ax[0].plot(forces['t'],forces['vB'],'g',label='B')
ax[0].set_ylabel(r'$\omega$')
ax[0].set_xticks([])
ax[0].legend()
ax[0].grid()
ax[0].set_xlim([tmin,tmax])

ax[1].plot(forces['t'],forces['Cl_F'],'m')
ax[1].plot(forces['t'],forces['Cl_T'],'b')
ax[1].plot(forces['t'],forces['Cl_B'],'g')
ax[1].set_ylabel(r'$C_l$')
ax[1].set_xticks([])
ax[1].grid()
ax[1].set_xlim([tmin,tmax])

ax[2].plot(forces['t'],forces['Cd_F'],'m')
ax[2].plot(forces['t'],forces['Cd_T'],'b')
ax[2].plot(forces['t'],forces['Cd_B'],'g')
ax[2].set_ylabel(r'$C_d$')
ax[2].set_xticks([])
ax[2].grid()
ax[2].set_xlim([tmin,tmax])

ax[3].plot(forces['t'],forces['Cd_total'],'k',label='$C_d$')
ax[3].plot(forces['t'],forces['Cl_total'],'r',label='$C_l$')
ax[3].set_ylabel(r'total force')
ax[3].set_xlabel(r'$t/\tau$')
ax[3].grid()
ax[3].legend()
ax[3].set_xlim([tmin,tmax])
plt.show()

fig, ax = plt.subplots(4,1)
ax[0].plot(forces['t'],forces['vT'],'b',label='T')
ax[0].set_ylabel(r'$\omega_T$')
ax[0].grid()
ax[0].set_xlabel(r'$t/\tau$')
ax[0].set_xlim([tmin,tmax])

t_delay = 0.5
x, y = forces['Cl_T'][int((tmin+t_delay)*10):int(tmax*10+1),0],forces['Cl_T'][int(tmin)*10:int((tmax-t_delay)*10+1),0]
t = forces['vT'][int((tmin+t_delay)*10):int(tmax*10+1),0]
ax[1].plot(t,x/y,'b',label='T')
ax[1].set_ylabel(r'$C_{l_T}(t/\tau+0.5)/C_{l_T}(t/\tau)$')
ax[1].set_xlabel(r'$t/\tau$')
ax[1].set_xlim([tmin,tmax])
ax[1].grid()

Cl_T_filt = savgol_filter(forces['Cl_T'][:,0],300,2)
ax[2].plot(forces['t'],Cl_T_filt,'b',label='T')
ax[2].set_ylabel(r'$C_{l_T}$')
ax[2].set_xlabel(r'$t/\tau$')
ax[2].set_xlim([tmin,tmax])
ax[2].grid()

ax[3].plot(forces['vT'][int((tmin)*10):int(tmax*10+1),0],forces['Cl_T'][int((tmin)*10):int(tmax*10+1),0],'b',label='T')
ax[3].set_ylabel(r'$C_{l_T}$')
ax[3].set_xlabel(r'$\omega_T$')
ax[3].grid()

plt.show()
a=0


fig, ax = plt.subplots(1,1)

ax.plot(forces['t'],Cl_T_filt)
ax.plot(forces['t'],forces['vT'])