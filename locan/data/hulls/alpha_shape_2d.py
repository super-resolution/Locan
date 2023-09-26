"""
Alpha shape utility functions for 2d.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__: list[str] = []


def _circumcircle(points: npt.ArrayLike, simplex: list[int]) -> tuple[float, float]:
    """
    Center and radius of circumcircle for one triangle.

    Parameters
    -----------
    points
        Point coordinates with shape (n_points, 2)
    simplex
        List with three indices representing a triangle from three points.

    Returns
    -------
    tuple[float, float]
        Center and radius of circumcircle
    """
    A = np.asarray(points)[simplex]
    M = np.array(
        [np.linalg.norm(A, axis=1) ** 2, A[:, 0], A[:, 1], np.ones(3)], dtype=np.float32
    )
    S = np.array(
        [0.5 * np.linalg.det(M[[0, 2, 3]]), -0.5 * np.linalg.det(M[[0, 1, 3]])]
    )
    a = np.linalg.det(M[1:])
    b = np.linalg.det(M[[0, 1, 2]])
    return S / a, np.sqrt(b / a + np.linalg.norm(S) ** 2 / a**2)  # center, radius


def _half_distance(points: npt.ArrayLike) -> npt.NDArray[np.float_]:
    """
    Half the distance between two points.

    Parameters
    -----------
    points
        point coordinates with shape (2, 2) representing a line

    Returns
    -------
    float
        Half the distance between the two points.
    """
    points = np.asarray(points)
    return_value: npt.NDArray[np.float_] = (
        np.sqrt((points[1, 0] - points[0, 0]) ** 2 + (points[1, 1] - points[0, 1]) ** 2)
        / 2
    )  # radius = _half_distance
    # this is faster than using: np.linalg.norm(np.diff(A, axis=0)) / 2
    return return_value
