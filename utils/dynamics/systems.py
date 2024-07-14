import numpy as np
from scipy.integrate import odeint

def get_lorenz_time(t_span, Y0, sigma, beta, rho, b_span = 0, flag_control = 0):

    def lorenz(y, t, sigma, beta, rho):

        u, v, w = y
        dydt = [-sigma*(u - v),
                rho*u - v - u*w,
                -beta*w + u*v]

        return dydt

    def lorenz_forced(y, t, sigma, beta, rho, t_span, b_span):

        u, v, w = y
        b = np.interp(t,t_span, b_span[:,0])

        dydt = [-sigma*(u - v) + b,
                rho*u - v - u*w,
                -beta*w + u*v]

        return dydt

    if flag_control:
        Y = odeint(lorenz_forced, Y0, t_span, args=(sigma, beta, rho, t_span, b_span))
    else:
        Y = odeint(lorenz, Y0, t_span, args=(sigma, beta, rho))

    return Y