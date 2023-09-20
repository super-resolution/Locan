"""

Transform localization data with a BUnwarpJ transformation matrix.

This module provides functions to transform coordinates in LocData objects by
applying a B-spline transformation as defined with the ImageJ/Fiji
plugin BunwarpJ_ [1]_, [2]_.

.. _BunwarpJ: https://imagej.net/BUnwarpJ

References
----------

.. [1] I. Arganda-Carreras, C. O. S. Sorzano, R. Marabini, J.-M. Carazo,
   C. Ortiz-de Solorzano, and J. Kybic,
   "Consistent and Elastic Registration of Histological Sections using
   Vector-Spline Regularization",
   Lecture Notes in Computer Science, Springer Berlin / Heidelberg,
   volume 4241/2006,
   CVAMIA: Computer Vision Approaches to Medical Image Analysis,
   pages 85-95, 2006.

.. [2] C.Ó. Sánchez Sorzano, P. Thévenaz, M. Unser,
   "Elastic Registration of Biological Images Using Vector-Spline
   Regularization",
   IEEE Transactions on Biomedical Engineering, vol. 52, no. 4,
   pages 652-663, 2005.

"""
from __future__ import annotations

import os
import sys
from itertools import islice
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import interpolate

from locan.data.locdata import LocData
from locan.data.metadata_utils import _modify_meta
from locan.data.transform.spatial_transformation import transform_affine

__all__: list[str] = ["bunwarp"]


def _unwarp(
    points: npt.ArrayLike,
    matrix_x: npt.ArrayLike,
    matrix_y: npt.ArrayLike,
    pixel_size: tuple[float, float],
) -> npt.NDArray[np.float_]:
    """
    Transform points with raw matrix from BunwarpJ.

    Parameters
    ----------
    points
        Point coordinates to be transformed.
        Array with shape (n_points, 2).
    matrix_x
        Transformation matrix for x coordinates
    matrix_y
        Transformation matrix for y coordinates
    pixel_size
        Pixel size for x and y component as used in ImageJ for registration

    Returns
    -------
    npt.NDArray[np.float_]
        Transformed point coordinates with shape (n_points, 2).
    """
    points_ = np.asarray(points)
    point_indices = np.divide(points_, pixel_size)

    matrix_x = np.asarray(matrix_x)
    matrix_y = np.asarray(matrix_y)
    if matrix_x.shape == matrix_y.shape:
        matrix_size = matrix_x.shape
    else:
        raise TypeError("matrix_x and matrix_y must have the same shape.")

    x = np.arange(matrix_size[0])
    y = np.arange(matrix_size[1])
    rgi_x = interpolate.RegularGridInterpolator(points=(x, y), values=matrix_x)
    rgi_y = interpolate.RegularGridInterpolator(points=(x, y), values=matrix_y)
    matrix_x_interpolated = rgi_x(point_indices, method="linear")
    matrix_y_interpolated = rgi_y(point_indices, method="linear")

    new_point_indices = np.stack([matrix_x_interpolated, matrix_y_interpolated], axis=1)
    new_points = np.multiply(new_point_indices, pixel_size)
    return new_points


def _read_matrix(
    path: str | os.PathLike[Any],
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Read file with raw matrix from BunwarpJ.

    Parameters
    ----------
    path
        Path to file with a raw matrix from BunwarpJ.

    Returns
    -------
    tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]
        x transformation array, y transformation array
    """
    with open(path) as file:
        header = list(islice(file, 2))

    # Get image heigth and width
    width = int(header[0].split("=")[1])
    height = int(header[1].split("=")[1])
    matrix_size = np.array([width, height])

    matrix_x = pd.read_csv(
        path, skiprows=4, header=None, nrows=matrix_size[1], delim_whitespace=True
    ).values.T  # transform values to get array[x, y]
    matrix_y = pd.read_csv(
        path, skiprows=(6 + matrix_size[1]), header=None, delim_whitespace=True
    ).values.T  # transform values to get array[x, y]

    return matrix_x, matrix_y


def bunwarp(
    locdata: LocData,
    matrix_path: str | os.PathLike[Any],
    pixel_size: tuple[float, float],
    flip: bool = False,
) -> LocData:
    """
    Transform coordinates by applying a B-spline transformation
    as represented by a raw transformation matrix from BunwarpJ.

    Parameters
    ----------
    locdata
        specifying the localization data on which to perform the manipulation.
    matrix_path
        Path to file with a raw matrix from BunwarpJ.
    pixel_size
        Pixel sizes used to determine transition matrix in ImageJ
    flip
        Flip locdata along x-axis before transformation

    Returns
    -------
    LocData
        New localization data with transformed coordinates.
    """
    local_parameter = locals()

    matrix_x, matrix_y = _read_matrix(matrix_path)

    if flip:
        image_size = np.multiply(matrix_x.shape, pixel_size)
        locdata = transform_affine(  # type: ignore[assignment]
            locdata, matrix=[[-1, 0], [0, 1]], offset=[image_size[0], 0]
        )

    new_points = _unwarp(locdata.coordinates, matrix_x, matrix_y, pixel_size)

    # new LocData object
    new_dataframe = locdata.data.copy()
    df = pd.DataFrame(
        {"position_x": new_points[:, 0], "position_y": new_points[:, 1]},
        index=locdata.data.index,
    )
    new_dataframe.update(df)
    new_locdata = LocData.from_dataframe(new_dataframe)

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=None,
    )
    new_locdata.meta = meta_

    return new_locdata
