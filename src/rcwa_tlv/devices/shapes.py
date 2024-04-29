
import numpy as np


def add_circle(array: np.ndarray, center: tuple, radius: float, val: float | complex) -> np.ndarray:
    """
    Add a circle with a given value to a 2D array.

    Parameters:
        array (numpy.ndarray): The 2D array.
        center (tuple): Coordinates of the center of the circle (row, column).
        radius (int): Radius of the circle.
        val (int or float): Value to add to the array within the circle.

    Returns:
        numpy.ndarray: The updated array.
    """
    rows, cols = np.indices(array.shape)
    row_center, col_center = center

    mask = (rows - row_center) ** 2 + (cols - col_center) ** 2 <= radius ** 2
    array[mask] = val

    return array


def add_rectangle(array: np.ndarray, top_left: tuple, bottom_right: tuple, val: float) -> np.ndarray:
    """
    Add a rectangle with a given value to a 2D array.

    Parameters:
        array (numpy.ndarray): The 2D array.
        top_left (tuple): Coordinates of the top-left corner of the rectangle (row, column).
        bottom_right (tuple): Coordinates of the bottom-right corner of the rectangle (row, column).
        val (int or float): Value to add to the array within the rectangle.

    Returns:
        numpy.ndarray: The updated array.
    """
    rows, cols = np.indices(array.shape)
    row_tl, col_tl = top_left
    row_br, col_br = bottom_right

    mask = (rows >= row_tl) & (rows <= row_br) & (cols >= col_tl) & (cols <= col_br)
    array[mask] = val

    return array


if __name__ == '__main__':
    n_height = 500
    n_width = 700

    layer = 3.0 * np.ones((n_height, n_width))

    layer = add_circle(layer, center=(0, 0), radius=17.7, val=1.0)
    layer = add_rectangle(layer, top_left=(8, 10), bottom_right=(19.7, 33.8), val=7.0)
