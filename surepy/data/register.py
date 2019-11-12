"""

Register localization data.

This module registers localization data and provides transformation parameters to put other localization data
in registry.

"""
import numpy as np

try:
    import open3d as o3d
    _has_open3d = True
except ImportError:
    _has_open3d = False

from surepy.data.locdata import LocData
from surepy.data.transform.transformation import _homogeneous_matrix


__all__ = ['register_icp']


def _register_icp_open3d(points, other_points, matrix=None, offset=None, pre_translation=None,
                         max_correspondence_distance=1_000, max_iteration=10_000, verbose=True):
    """
    Register `points` by an "Iterative Closest Point" algorithm using open3d.

    Parameters
    ----------
    points : array-like
        Points representing the source on which to perform the manipulation.
    other_points : array-like
        Points representing the target.
    matrix : tuple with shape (d, d)
        Transformation matrix used as initial value. If None the unit matrix is used.
    offset : tuple of int or float with shape (d,)
        Translation vector used as initial value. If None a vector of zeros is used.
    pre_translation : tuple of int or float
        Values for translation of coordinates before registration.
    max_correspondence_distance : float
        Threshold distance for the icp algorithm. Parameter is passed to open3d algorithm.
    max_iteration : int
        Maximum number of iterations. Parameter is passed to open3d algorithm.
    verbose : bool
        Flag indicating if transformation results are printed out.

    Returns
    -------
    tuple of ndarrays
        Matrix and offset representing the optimized transformation.
    """
    if not _has_open3d:
        raise ImportError("open3d is required.")

    points_ = np.asarray(points)
    other_points_ = np.asarray(other_points)

    # prepare 3d
    if np.shape(points)[1] == np.shape(other_points)[1]:
        dimension = np.shape(points)[1]
    else:
        raise ValueError('Dimensions for locdata and other_locdata are incompatible.')

    if dimension == 2:
        points_3d = np.concatenate([points_, np.zeros((len(points_), 1))], axis=1)
        other_points_3d = np.concatenate([other_points_, np.zeros((len(other_points_), 1))], axis=1)
    elif dimension == 3:
        points_3d = points_
        other_points_3d = other_points_
    else:
        raise ValueError('Point array has the wrong shape.')

    # points in open3d
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)

    other_point_cloud = o3d.geometry.PointCloud()
    other_point_cloud.points = o3d.utility.Vector3dVector(other_points_3d)

    # initial matrix
    matrix_ = np.identity(dimension) if matrix is None else matrix
    offset_ = np.zeros(dimension) if offset is None else offset

    matrix_3d = np.identity(3)
    matrix_3d[:dimension, :dimension] = matrix_
    offset_3d = np.zeros(3)
    offset_3d[:dimension] = offset_
    matrix_homogeneous = _homogeneous_matrix(matrix_3d, offset_3d)

    if pre_translation is not None:
        pre_translation_3d = np.zeros(3)
        pre_translation_3d[:dimension] = pre_translation
        other_point_cloud.translate(pre_translation_3d)

    # apply ICP
    registration = o3d.registration.registration_icp(
        source=point_cloud, target=other_point_cloud,
        max_correspondence_distance=max_correspondence_distance,
        init=matrix_homogeneous,
        estimation_method=o3d.registration.TransformationEstimationPointToPoint(with_scaling=True),
        criteria=o3d.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )

    if dimension == 2:
        new_matrix = registration.transformation[0:2, 0:2]
        new_offset = registration.transformation[0:2, 3]
    else:  # if dimension == 3:
        new_matrix = registration.transformation[0:3, 0:3]
        new_offset = registration.transformation[0:3, 3]

    if verbose:
        print(registration)

    return new_matrix, new_offset


def register_icp(locdata, other_locdata, matrix=None, offset=None, pre_translation=None,
                 max_correspondence_distance=1_000, max_iteration=10_000, verbose=True):
    """
    Register `points` or coordinates in `locdata` by an "Iterative Closest Point" algorithm using open3d.

    Parameters
    ----------
    locdata : array-like or LocData object
        Localization data representing the source on which to perform the manipulation.
    other_locdata : array-like or LocData object
        Localization data representing the target.
    matrix : tuple with shape (d, d)
        Transformation matrix used as initial value. If None the unit matrix is used.
    offset : tuple of int or float with shape (d,)
        Translation vector used as initial value. If None a vector of zeros is used.
    pre_translation : tuple of int or float
        Values for translation of coordinates before registration.
    max_correspondence_distance : float
        Threshold distance for the icp algorithm. Parameter is passed to open3d algorithm.
    max_iteration : int
        Maximum number of iterations. Parameter is passed to open3d algorithm.
    verbose : bool
        Flag indicating if transformation results are printed out.

    Returns
    -------
    tuple of ndarrays
        Matrix and offset representing the optimized transformation.
    """
    local_parameter = locals()

    # adjust input
    if isinstance(locdata, LocData):
        points = locdata.coordinates
    else:
        points = locdata

    if isinstance(other_locdata, LocData):
        other_points = other_locdata.coordinates
    else:
        other_points = other_locdata

    new_matrix, new_offset = _register_icp_open3d(points, other_points, matrix=matrix, offset=offset,
                                                  pre_translation=pre_translation,
                                                  max_correspondence_distance=1_000, max_iteration=10_000,
                                                  verbose=True)
    return new_matrix, new_offset
