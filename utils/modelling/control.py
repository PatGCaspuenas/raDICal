import numpy as np
import math

def rotation2coords(x, y, x0, y0, v, Dt, R=0.5):

    # ANGULAR VELOCITY
    w = v / R

    # ROTATION ANGLE
    Dtheta = w * Dt
    theta = math.atan2(y - y0, x - x0)
    theta_new = theta + Dtheta

    # POLAR TO CARTESIAN
    x_new = R * math.cos(theta_new) + x0
    y_new = R * math.sin(theta_new) + y0

    return x_new, y_new
