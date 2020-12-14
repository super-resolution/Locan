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
import sys
from itertools import islice

import numpy as np
import pandas as pd
from scipy import interpolate
from numba import jit

from surepy.data.locdata import LocData
from surepy.data.metadata_utils import _modify_meta


__all__ = ['bunwarp']


def _unwarp(points, matrix_x, matrix_y, pixel_size, matrix_size):
    """
    Transform points with raw matrix from BunwarpJ.

    Parameters
    ----------
    points : array-like with shape (n_points, 2)
        Point coordinates to be transformed
    matrix_x, matrix_y : array-like
        Transformation matrix for x and y coordinates
    pixel_size : tuple (2,)
        Pixel size for x and y component as used in ImageJ for registration
    matrix_size : tuple (2,)
        Number of matrix entries in each dimension

    Returns
    -------
    numpy.ndarray with shape (n_points, 2)
        Transformed point coordinates
    """
    points_ = np.asarray(points)
    point_indices = np.divide(points_, pixel_size)

    x = np.arange(matrix_size[0])
    y = np.arange(matrix_size[1])
    z_x = matrix_x
    f_x = interpolate.interp2d(x, y, z_x, kind='linear')
    z_y = matrix_y
    f_y = interpolate.interp2d(x, y, z_y, kind='linear')

    new_points = np.array(
        [point - np.concatenate((f_x(*pind), f_y(*pind))) for pind, point in zip(point_indices, points_)])
    return new_points


def _read_matrix(path):
    """
    Read file with raw matrix from BunwarpJ.

    Parameters
    ----------
    path : str or os.PathLike
        Path to file with a raw matrix from BunwarpJ.

    Returns
    -------
    tuple
        matrix width and height
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
    matrix_size = np.array([width, height])

    matrix_x = pd.read_csv(path, skiprows=4, header=None, nrows=matrix_size[1],
                                 delim_whitespace=True).values.T      # transform values to get array[x, y]
    matrix_y = pd.read_csv(path, skiprows=(6 + matrix_size[1]), header=None,
                                 delim_whitespace=True).values.T      # transform values to get array[x, y]

    return matrix_size, matrix_x, matrix_y


def bunwarp(locdata, matrix_path, pixel_size):
    """
    Transform coordinates by applying a B-spline transformation
    as represented by a raw transformation matrix from BunwarpJ.

    Parameters
    ----------
    locdata : LocData
        specifying the localization data on which to perform the manipulation.
    matrix_path :
        Path to file with a raw matrix from BunwarpJ.
    pixel_size : tuple
        Pixel sizes used to determine transition matrix in ImageJ

    Returns
    -------
    locdata : LocData
        New localization data with transformed coordinates.
    """
    local_parameter = locals()

    matrix_size, matrix_x, matrix_y = _read_matrix(matrix_path)
    new_points = _unwarp(locdata.coordinates, matrix_x, matrix_y, pixel_size, matrix_size)

    df = pd.DataFrame({'position_x': new_points[:, 0], 'position_y': new_points[:, 1]})
    new_locdata = LocData.from_dataframe(df)

    # update metadata
    meta_ = _modify_meta(locdata, new_locdata, function_name=sys._getframe().f_code.co_name,
                         parameter=local_parameter, meta=None)
    new_locdata.meta = meta_

    return new_locdata
