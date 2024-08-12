# PACKAGES
import numpy as np
import math

def rotation2coords(x, y, x0, y0, v, Dt, R=0.5):
    """
    Estimates for all cylinders the rotation coordinates along their surface after a given delta t

    :param x,y: cartesian coordinates of specific surface coordinate of cylinder(s) (at r = R)
    :param x0,y0: cartesian coordinates of cylinder centers
    :param v: linear velocity of cylinder surface (at r = R)
    :param Dt: delta time
    :param R: cylinder radius
    :return: cartesian coordinates of cylinder coordinates at r = R after rotation
    """

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
