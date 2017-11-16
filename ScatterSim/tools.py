''' Some tools for various

'''

import numpy as np
def rotmat3D(phi,axis=3):
    '''3D rotation matrix about z axis.

        phi : in degrees

        Counter-clockwise rotation is positive.
        axis: choose either:
            1 - x axis
            2 - y axis
            3 - z axis
    '''
    phi = np.radians(phi)
    if axis == 3:
        return np.array([
            [np.cos(phi), np.sin(phi),0, 0],
            [-np.sin(phi), np.cos(phi),0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 2:
        return np.array([
            [np.cos(phi), 0, np.sin(phi), 0],
            [0, 1, 0, 0],
            [-np.sin(phi), 0, np.cos(phi), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 1:
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), np.sin(phi), 0],
            [0, -np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1]
        ])
    else:
        print("Error, not a good axis specified. Specified: {}".format(axis))

