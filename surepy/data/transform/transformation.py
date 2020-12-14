"""

Transform localization data.

This module takes localization data and applies transformation procedures on coordinates or other properties.

"""
import sys

import numpy as np
import pandas as pd

from surepy.data.locdata import LocData
import surepy.data.hulls
from surepy.data.region import RoiRegion
from surepy.simulation import simulate_csr, simulate_csr_on_region
from surepy.data.metadata_utils import _modify_meta
from surepy.constants import _has_open3d
if _has_open3d: import open3d as o3d


__all__ = ['transform_affine', 'randomize']


def _transform_affine_numpy(points, matrix=None, offset=None, pre_translation=None):
    """
    Transform `points` by an affine transformation using standard numpy procedures.

    Parameters
    ----------
    points : array-like
        Points on which to perform the manipulation.
    matrix : tuple with shape (d, d)
        Transformation matrix. If None the unit matrix is used.
    offset : tuple of int or float with shape (d,)
        Translation vector. If None a vector of zeros is used.
    pre_translation : tuple of int or float
        Translation vector for coordinates applied before affine transformation.
        The reverse translation is applied after the affine transformation.

    Returns
    -------
    numpy.ndarray
        Transformed coordinates.
    """
    points_ = np.asarray(points)
    dimension = np.shape(points_)[1]

    matrix_ = np.identity(dimension) if matrix is None else matrix
    offset_ = np.zeros(dimension) if offset is None else offset

    if pre_translation is None:
        # transformed_points = np.array([np.dot(matrix_, point) + offset_ for point in points_])
        # same function but better performance:
        transformed_points = np.einsum("ij, nj -> ni", matrix_, points) + offset_
    else:
        transformed_points = points_ + pre_translation
        # transformed_points = np.array([np.dot(matrix_, point) + offset_ for point in transformed_points])
        transformed_points = np.einsum("ij, nj -> ni", matrix_, transformed_points) + offset_
        transformed_points = transformed_points - pre_translation

    return transformed_points


def _homogeneous_matrix(matrix=np.identity(3), offset=np.zeros(3)):
    """
    Combine transformation matrix and translation vector for dimension d into homogeneous (d+1, d+1) transformation
    matrix to be used with homogeneous coordinates.

    Parameters
    ----------
    matrix : array-like of int or float with shape (d, d)
        Transformation matrix.
    offset : array-like of int or float with shape (d,)
        Translation vector.

    Returns
    -------
    numpy.ndarray with shape (d+1, d+1)
        Homogeneous transformation matrix to be used with homogeneous coordinate vector.
    """
    dimension = np.shape(matrix)[0]

    if dimension != np.shape(matrix)[1]:
        raise ValueError('The matrix has to be of shape (d, d).')

    if dimension != np.shape(offset)[0]:
        raise ValueError('The matrix and offset must have the same dimension.')

    matrix_ = np.identity(dimension+1)
    matrix_[0:dimension, 0:dimension] = matrix
    matrix_[:dimension, dimension] = offset

    return matrix_


def _transform_affine_open3d(points, matrix=None, offset=None, pre_translation=None):
    """
    Transform `points` or coordinates in `locdata` by an affine
    transformation using open3d.

    Parameters
    ----------
    points : array-like
        Points on which to perform the manipulation.
    matrix : tuple with shape (d, d)
        Transformation matrix. If None the unit matrix is used.
    offset : tuple of int or float with shape (d,)
        Translation vector. If None a vector of zeros is used.
    pre_translation : tuple of int or float
        Translation vector for coordinates applied before affine transformation.
        The reverse translation is applied after the affine transformation.

    Returns
    -------
    numpy.ndarray
        Transformed coordinates.
    """
    if not _has_open3d:
        raise ImportError("open3d is required.")

    points_ = np.asarray(points)
    dimension = np.shape(points_)[1]

    matrix_ = np.identity(dimension) if matrix is None else matrix
    offset_ = np.zeros(dimension) if offset is None else offset

    # prepare 3d
    if dimension == 2:
        points_3d = np.concatenate([points_, np.zeros((len(points_), 1))], axis=1)
    elif dimension == 3:
        points_3d = points_
    else:
        raise ValueError('Point array has the wrong shape.')

    # points in open3d
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)

    # transform
    matrix_3d = np.identity(3)
    matrix_3d[:dimension, :dimension] = matrix_
    offset_3d = np.zeros(3)
    offset_3d[:dimension] = offset_
    matrix_homogeneous = _homogeneous_matrix(matrix_3d, offset_3d)

    if pre_translation is None:
        point_cloud.transform(matrix_homogeneous)
    else:
        pre_translation_3d = np.zeros(3)
        pre_translation_3d[:dimension] = pre_translation
        point_cloud.translate(pre_translation_3d)
        point_cloud.transform(matrix_homogeneous)
        point_cloud.translate(-pre_translation_3d)

    # return correct dimension
    if dimension == 2:
        transformed_points = np.asarray(point_cloud.points)[:, 0:2]
    else:  # if dimension == 3:
        transformed_points = np.asarray(point_cloud.points)

    return transformed_points


def transform_affine(locdata, matrix=None, offset=None, pre_translation=None, method='numpy'):
    """
    Transform `points` or coordinates in `locdata` by an affine transformation.

    Parameters
    ----------
    locdata : numpy.ndarray or LocData
        Localization data on which to perform the manipulation.
    matrix :
        Transformation matrix. If None the unit matrix is used.
    offset : tuple of int or float
        Values for translation. If None a vector of zeros is used.
    pre_translation : tuple of int or float
        Translation vector for coordinates applied before affine transformation.
        The reverse translation is applied after the affine transformation.
    method : string
        The method (i.e. library or algorithm) used for computation. One of 'numpy', 'open3d'.

    Returns
    -------
    numpy.ndarray or LocData
        New localization data with transformed coordinates.
    """
    local_parameter = locals()

    if len(locdata) == 0:
        return locdata

    # adjust input
    if isinstance(locdata, LocData):
        points = locdata.coordinates
    else:
        points = locdata

    if method == 'numpy':
        transformed_points = _transform_affine_numpy(points, matrix=matrix, offset=offset,
                                                     pre_translation=pre_translation)
    elif method == 'open3d':
        transformed_points = _transform_affine_open3d(points, matrix=matrix, offset=offset,
                                                      pre_translation=pre_translation)
    else:
        raise ValueError(f'Method {method} is not available.')

    # prepare output
    if isinstance(locdata, LocData):
        # new LocData object
        new_dataframe = locdata.data.copy()
        new_dataframe.update(pd.DataFrame(transformed_points, columns=locdata.coordinate_labels,
                                          index=locdata.data.index))
        new_locdata = LocData.from_dataframe(new_dataframe)

        # update metadata
        meta_ = _modify_meta(locdata, new_locdata, function_name=sys._getframe().f_code.co_name,
                             parameter=local_parameter,
                             meta=None)
        new_locdata.meta = meta_

        return new_locdata

    else:
        return transformed_points


def randomize(locdata, hull_region='bb'):
    """
    Transform locdata coordinates into randomized coordinates that follow complete spatial randomness on the same
    region as the input locdata.

    Parameters
    ----------
    locdata : LocData
        Localization data to be randomized
    hull_region : str, RoiRegion Object, or dict
        Region of interest as specified by a hull or a `RoiRegion` or dictionary with keys `region_specs` and
        `region_type`.
        Allowed values for `region_specs` and `region_type` are defined in the docstrings for `Roi` and `RoiRegion`.
        String identifier can be one of 'bb', 'ch', 'as', 'obb' referring to the corresponding hull.

    Returns
    -------
    locdata : LocData
        New localization data with randomized coordinates.
    """
    local_parameter = locals()

    if hull_region == 'bb':
        try:
            ranges = locdata.bounding_box.hull.T
        except AttributeError:
            locdata.bounding_box = surepy.data.hulls.BoundingBox(locdata.coordinates)
            ranges = locdata.bounding_box.hull.T

        new_locdata = simulate_csr(n_samples=len(locdata), n_features=len(ranges), feature_range=ranges)

    elif hull_region == 'ch':
        try:
            region_ = locdata.convex_hull.region
        except AttributeError:
            region_ = surepy.data.hulls.ConvexHull(locdata.coordinates).region

        new_locdata = simulate_csr_on_region(region_, n_samples=len(locdata))

    # todo: implement simulate_csr in 3D
    # todo: implement simulate_csr on as and obb hull regions

    elif isinstance(hull_region, (RoiRegion, dict)):
        if isinstance(hull_region, dict):
            region_ = RoiRegion(region_specs=hull_region['region_specs'], region_type=hull_region['region_type'])
        else:
            region_ = hull_region

        new_locdata = simulate_csr_on_region(region_, n_samples=len(locdata))

    else:
        raise NotImplementedError

    # update metadata
    meta_ = _modify_meta(locdata, new_locdata, function_name=sys._getframe().f_code.co_name,
                         parameter=local_parameter, meta=None)
    new_locdata.meta = meta_

    return new_locdata
