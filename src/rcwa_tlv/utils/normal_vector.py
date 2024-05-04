
import numpy as np


def calc_edges_diffs(img, axis=1):

    if axis == 0:
        nv = np.roll(img, shift=1, axis=0) - img
        nv += img - np.roll(img, shift=-1, axis=0)
    elif axis == 1:
        nv = np.roll(img, shift=1, axis=1) - img
        nv += img - np.roll(img, shift=-1, axis=1)
    else:
        raise ValueError('axis must be 0 or 1')

    return nv


def calc_nv_fields(layer):
    abs_layer = np.abs(layer)
    nv_x = calc_edges_diffs(abs_layer, axis=1)
    nv_y = calc_edges_diffs(abs_layer, axis=0)

    norm = np.sqrt(nv_x ** 2 + nv_y ** 2)
    nv_x = np.where(np.abs(nv_x) > 0, nv_x / norm, 0.)
    nv_y = np.where(np.abs(nv_y) > 0, nv_y / norm, 0.)

    return nv_x, nv_y
