"""
Functions to compute locdata properties.
"""
from collections import namedtuple

import numpy as np
from scipy.spatial.distance import pdist


__all__ = ['max_distance', 'compute_inertia_moments']


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


def compute_inertia_moments(points):
    """
    Return inertia moments (or principal components) and related properties for the given points.
    Inertia moments are represented by eigenvalues (and corresponding eigenvectors) of the covariance matrix.
    Variance_explained represents the eigenvalues normalilzed to the sum of all eigenvalues.
    For 2-dimensional data, orientation is the angle (in degrees) between the principal axis with the largest variance and the x-axis.
    Also for 2-dimensional data, excentricity is computed as e=Sqrt(1-M_min/M_max).

    Parameters
    ----------
    points : tuple, list, numpy.ndarray of shape (n_points, n_dim)
        Point coordinates or other data.

    Returns
    -------
    namedtuple(InertiaMoments, 'eigenvalues eigenvectors variance_explained orientation excentricity')
        A tuple with eigenvalues, eigenvectors, variance_explained, orientation, excentricity

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
        excentricity = np.sqrt(1 - np.min(eigen_values) / np.max(eigen_values))
    else:  # todo implement for 3d
        orientation = np.nan
        excentricity = np.nan

    InertiaMoments = namedtuple('InertiaMoments',
                                'eigenvalues eigenvectors variance_explained orientation excentricity')
    inertia_moments = InertiaMoments(eigen_values, eigen_vectors, variance_explained, orientation, excentricity)
    return inertia_moments
