"""
Alpha shape utility functions for 2d
"""
import numpy as np


__all__ = []


def _circumcircle(points, simplex):
    """
    Center and radius of circumcircle for one triangle.

    Parameters
    -----------
    points : array of shape (n_points, 2)
        point coordinates
    simplex : list
        list with three indices representing a triangle from three points.

    Returns
    -------
    tuple of float
        Center and radius of circumcircle
    """
    A = np.asarray(points)[simplex]
    M = np.array([np.linalg.norm(A, axis=1)**2, A[:, 0], A[:, 1], np.ones(3)], dtype=np.float32)
    S = np.array([0.5 * np.linalg.det(M[[0, 2, 3]]), -0.5 * np.linalg.det(M[[0, 1, 3]])])
    a = np.linalg.det(M[1:])
    b = np.linalg.det(M[[0, 1, 2]])
    return S / a, np.sqrt(b / a + np.linalg.norm(S)**2 / a**2)  # center, radius


def _half_distance(points):
    """
    Half the distance between two points.

    Parameters
    -----------
    points : array of shape (2, 2)
        point coordinates representing a line

    Returns
    -------
    float
        Half the distance between the two points.
    """
    points = np.asarray(points)
    return np.sqrt((points[1, 0] - points[0, 0])**2 + (points[1, 1] - points[0, 1])**2) / 2  # radius = _half_distance
    # this is faster than using: np.linalg.norm(np.diff(A, axis=0)) / 2
