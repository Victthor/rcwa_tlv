
import numpy as np
from scipy.linalg import toeplitz


def convmat(matrix_in, p, q=None, dtype=np.complex128):
    """
    title: convmat Rectangular Convolution Matrix
    This function constructs convolution matrices from a real space grid.

    input:
        matrix_in - input matrix
        p - number of spatial harmonics along x
        q - number of spatial harmonics along y (optional, default is 1)

    output:
        conv_matrix - convolution matrix

    notes:
        WARNING!!! Works properly only when the number of harmonics is less than half number
        of points in a particular dimension

        C = convmat(A, P)       for 1D problems
        C = convmat(A, P, Q)    for 2D problems
    """
    # DETERMINE SIZE OF A
    ny, nx = matrix_in.shape

    # HANDLE NUMBER OF HARMONICS FOR ALL DIMENSIONS
    q = q if q is not None else 1

    # COMPUTE FOURIER COEFFICIENTS OF A
    matrix_in_fft = (np.fft.fftshift(np.fft.fftn(matrix_in)) / (nx * ny)).astype(dtype)

    # COMPUTE ARRAY INDICES OF CENTER HARMONIC
    p0 = (nx // 2)
    q0 = (ny // 2)

    # x domain:
    x = np.arange(p0, p0 - p, -1)
    x = np.tile(x[np.newaxis, :], (1, q))
    xp = np.arange(p)
    x = x + xp[:, np.newaxis]
    x = np.tile(x, (q, 1))

    # y domain:
    a = np.arange(q)
    y = toeplitz(a, -a)
    y = np.repeat(np.repeat(y, p, axis=0), p, axis=1) + q0

    conv_matrix = matrix_in_fft[y, x]

    return conv_matrix
