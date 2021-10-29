"""
Functions to compute locdata properties.
"""
from collections import namedtuple
import warnings
import logging

import numpy as np
from scipy.spatial.distance import pdist
from shapely.geometry import Point

from locan.data.region import Region, Region2D, RoiRegion


__all__ = ['distance_to_region', 'distance_to_region_boundary', 'max_distance', 'inertia_moments']

logger = logging.getLogger(__name__)


def distance_to_region(locdata, region):
    """
    Determine the distance to the nearest point within `region` for all localizations.
    Returns zero if localization is within the region.

    Parameters
    ----------
    locdata : LocData
        Localizations for which distances are determined.

    region : Region
        Region from which the closest point is selected.

    Returns
    --------
    numpy.ndarray
        Distance for each localization.
    """
    distances = np.full(len(locdata), 0.)
    if isinstance(region, (Region2D, RoiRegion)):
        for i, point in enumerate(locdata.coordinates):
            distances[i] = Point(point).distance(region.shapely_object)
    else:
        raise NotImplementedError("Region must be Region2D object.")

    return distances


def distance_to_region_boundary(locdata, region):
    """
    Determine the distance to the nearest region boundary for all localizations.
    Returns a positive value regardless of weather the point is within or outside the region.

    Parameters
    ----------
    locdata : LocData
        Localizations for which distances are determined.

    region : Region
        Region from which the closest point is selected.

    Returns
    --------
    numpy.ndarray
        Distance for each localization.
    """
    distances = np.full(len(locdata), 0.)
    if isinstance(region, (Region2D, RoiRegion)):
        for i, point in enumerate(locdata.coordinates):
            distances[i] = Point(point).distance(region.shapely_object.boundary)
    else:
        raise NotImplementedError("Region must be Region2D object.")

    return distances


def max_distance(locdata):
    """
    Return maximum of all distances between any two localizations in locdata.

    Parameters
    ----------
    locdata : LocData
        Localization data

    Returns
    -------
    dict
        A dict with key `max_distance` and the corresponding value being the maximum distance.
    """
    points = locdata.convex_hull.vertices
    D = pdist(points)
    distance = np.nanmax(D)
    return {'max_distance': distance}


def inertia_moments(points):
    """
    Return inertia moments (or principal components) and related properties for the given points.
    Inertia moments are represented by eigenvalues (and corresponding eigenvectors) of the covariance matrix.
    Variance_explained represents the eigenvalues normalized to the sum of all eigenvalues.
    For 2-dimensional data, orientation is the angle (in degrees) between the principal axis
    with the largest variance and the x-axis.
    Also for 2-dimensional data, eccentricity is computed as e=Sqrt(1-M_min/M_max).

    Parameters
    ----------
    points : tuple, list, numpy.ndarray of shape (n_points, n_dim)
        Point coordinates or other data.

    Returns
    -------
    namedtuple(InertiaMoments, 'eigenvalues eigenvectors variance_explained orientation eccentricity')
        A tuple with eigenvalues, eigenvectors, variance_explained, orientation, eccentricity

    Note
    ----
    The data is not standardized.
    """
    points = np.asarray(points)
    covariance_matrix = np.cov(points.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    variance_explained = [eigen_value / sum(eigen_values) for eigen_value in eigen_values]
    index_max_eigen_value = np.argmax(eigen_values)

    if np.shape(points)[1] == 2:
        orientation = np.degrees(
            np.arctan2(eigen_vectors[1][index_max_eigen_value], eigen_vectors[0][index_max_eigen_value]))
        eccentricity = np.sqrt(1 - np.min(eigen_values) / np.max(eigen_values))
    else:  # todo implement for 3d
        logger.warning("Orientation and eccentricity have not yet been implemented for 3D.")
        orientation = np.nan
        eccentricity = np.nan

    InertiaMoments = namedtuple('InertiaMoments',
                                'eigenvalues eigenvectors variance_explained orientation eccentricity')
    inertia_moments = InertiaMoments(eigen_values, eigen_vectors, variance_explained, orientation, eccentricity)
    return inertia_moments
