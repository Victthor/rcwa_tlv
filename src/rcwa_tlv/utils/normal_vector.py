
import numpy as np


def calc_edges_diffs(img, axis=1):

    diff = np.diff(img, axis=axis)
    nv = np.zeros_like(img, dtype=np.float32)
    vectors = np.where(diff == 0, 0, 1 / np.sqrt(2))

    if axis == 0:
        nv[:-1, :] += vectors
        nv[1:, :] += vectors
    elif axis == 1:
        nv[:, :-1] += vectors
        nv[:, 1:] += vectors
    else:
        raise ValueError('axis must be 0 or 1')

    return nv


def calc_nv_fields(layer):
    abs_layer = np.abs(layer)
    nv_x = calc_edges_diffs(abs_layer, axis=1)
    nv_y = calc_edges_diffs(abs_layer, axis=0)

    return nv_x, nv_y
