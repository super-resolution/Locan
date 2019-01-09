"""

Transform localization data with a BUnwarpJ transformation matrix.

This module provides functions to transform coordinates in LocData objects by applying a B-spline transformation as
defined with the ImageJ/Fiji plugin BunwarpJ_ [1]_, [2]_.

.. _BunwarpJ: https://imagej.net/BUnwarpJ

References
----------

.. [1] I. Arganda-Carreras, C. O. S. Sorzano, R. Marabini, J.-M. Carazo, C. Ortiz-de Solorzano, and J. Kybic,
   "Consistent and Elastic Registration of Histological Sections using Vector-Spline Regularization",
   Lecture Notes in Computer Science, Springer Berlin / Heidelberg, volume 4241/2006,
   CVAMIA: Computer Vision Approaches to Medical Image Analysis, pages 85-95, 2006.

.. [2] C.Ó. Sánchez Sorzano, P. Thévenaz, M. Unser,
   "Elastic Registration of Biological Images Using Vector-Spline Regularization",
   IEEE Transactions on Biomedical Engineering, vol. 52, no. 4, pp. 652-663, April 2005.

"""

import numpy as np
import pandas as pd
from itertools import islice
from numba import jit

from surepy import LocData


@jit(nopython=True)
def _unwarp(loc_input_1, matrix_X, matrix_Y, pixel_size, real_size):
    """
    Apply transformation.
    """
    for i in range(loc_input_1.shape[0]):  # drauf achten das x der erste Eintrag und Y der zweite Eintrag ist

        x_1 = (loc_input_1[i][0] * 10 ** -9 / real_size[0]) * pixel_size[
            0] - 0.5  # -1 wegen array indices pixelsize=x arraysize=x-1 startet bei 0+0.5 pixel offset
        y_1 = (loc_input_1[i][1] * 10 ** -9 / real_size[1]) * pixel_size[1] - 0.5
        x_1_G = int(x_1)
        y_1_G = int(y_1)
        x_1_N = x_1 - x_1_G
        y_1_N = y_1 - y_1_G
        x_2_Ganz = matrix_X[y_1_G, x_1_G]
        y_2_Ganz = matrix_Y[y_1_G, x_1_G]
        if (x_1_G < (pixel_size[0] - 1)) and (y_1_G < (pixel_size[1] - 1)):
            x_2 = x_2_Ganz + (matrix_X[y_1_G + 1, x_1_G] - x_2_Ganz) * x_1_N + (matrix_X[
                                                                                    y_1_G, x_1_G + 1] - x_2_Ganz) * y_1_N  # nur interpolation fuer x interpolation fuer y fehlt. ; interpolation zeile
            y_2 = y_2_Ganz + (matrix_Y[y_1_G, x_1_G + 1] - y_2_Ganz) * y_1_N + (
                        matrix_Y[y_1_G + 1, x_1_G] - y_2_Ganz) * x_1_N
        else:
            x_2 = x_2_Ganz + (x_2_Ganz - matrix_X[y_1_G - 1, x_1_G]) * x_1_N + (
                        x_2_Ganz - matrix_X[y_1_G, x_1_G - 1]) * y_1_N
            y_2 = y_2_Ganz + (y_2_Ganz - matrix_Y[y_1_G, x_1_G - 1]) * y_1_N + (
                        y_2_Ganz - matrix_Y[y_1_G - 1, x_1_G]) * x_1_N  # interpolation spalte
        y_3 = (y_2 / pixel_size[1]) * real_size[1] * 10 ** 9
        x_3 = (x_2 / pixel_size[0]) * real_size[0] * 10 ** 9
        loc_input_1[i][0] = x_3
        loc_input_1[i][1] = y_3
    x_min = loc_input_1[:, 0].min()
    y_min = loc_input_1[:, 1].min()
    if x_min < 0:
        loc_input_1[:, 0] = loc_input_1[:, 0] + (x_min * -1)
    if y_min < 0:
        loc_input_1[:, 1] = loc_input_1[:, 1] + (y_min * -1)
    return loc_input_1


def _read_matrix(path):
    """
    Read file with raw matrix from BunwarpJ.

    Parameters
    ----------
    path : path object
        Path to file with a raw matrix from BunwarpJ.

    Returns
    -------
    tuple
        image height and width
    array
        x transformation array
    array
        y transformation array
    """
    with open(path) as file:
        header = list(islice(file, 2))

    # Get image heigth and width
    width = int(header[0].split("=")[1])
    height = int(header[1].split("=")[1])
    trans_size = np.array([width, height])

    trans_matrix_x = pd.read_csv(path, skiprows=4, header=None, nrows=trans_size[1],
                                 delim_whitespace=True).as_matrix()
    trans_matrix_y = pd.read_csv(path, skiprows=(6 + trans_size[1]), header=None,
                                 delim_whitespace=True).as_matrix()

    return (trans_size, trans_matrix_x, trans_matrix_y)


def bunwarp(locdata, matrix_path):
    """
    Transform coordinates by applying a B-spline transformation
    as represented by a raw transformation matrix from BunwarpJ.

    Parameters
    ----------
    locdata : LocData object
        specifying the localization data on which to perform the manipulation.
    matrix_path :
        Path to file with a raw matrix from BunwarpJ.

    Returns
    -------
    locdata : LocData object
        New localization data with transformed coordinates.
    """
    (trans_size, trans_matrix_x, trans_matrix_y) = _read_matrix(matrix_path)
    loc_max = np.max(locdata.coordinates, axis=0)
    trans_array = _unwarp(locdata.coordinates.T, trans_matrix_x, trans_matrix_y, trans_size, loc_max)

    df = pd.DataFrame({'Position_x': trans_array[0], 'Position_y': trans_array[1]})

    # todo: add meta data
    new_locdata = LocData.from_dataframe(df)

    return new_locdata