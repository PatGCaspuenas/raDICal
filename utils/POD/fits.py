import numpy as np

def energy_truncation(S, E):

    # CUMULATIVE ENERGY
    energy = np.cumsum(S**2)/np.sum(S**2)

    r = np.where(energy >= E)[0][0]

    return r + 1 # NUMBER OF MODES (not index)

def elbow_fit(x, y):

    # PARAMETERS AND INITIALIZATION
    ny = len(y)
    R2 = np.zeros(ny)
    err = np.zeros(ny)

    for i in range(ny):

        # FIT FIRST ORDER POLYNOMIALS FOR CURVES SPLIT AT i
        if i == 0:
            y1 = y[0:(i+1)]
        else:
            c1 = np.polyfit(x[0:(i+1)], y[0:(i+1)], 1)
            y1 = np.polyval(c1, x[0:(i+1)])
        if i == ny-1:
            y2 = y[(i+1):]
        else:
            c2 = np.polyfit(x[(i+1):], y[(i+1):], 1)
            y2 = np.polyval(c2, x[(i+1):])

        yt = np.concatenate((y1, y2)).reshape(ny, 1)

        # COMPUTE R2 AND ERROR OF FITS
        R2[i] = 1 - np.sum((yt - y.reshape(ny, 1))**2) / np.sum((y.reshape(ny, 1) - np.mean(y.reshape(ny, 1)))**2)
        err[i] = np.sqrt(np.sum((yt- y.reshape(ny, 1))**2) / ny)

    # FIND THE ELBOW
    elbow = np.argmax(R2/err)

    return x[elbow] # NUMBER OF MODES (not index)