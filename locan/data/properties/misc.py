"""
Functions to compute locdata properties.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist
from shapely.geometry import Point

from locan.data.region import Region2D, RoiRegion

if TYPE_CHECKING:
    from locan.data.locdata import LocData
    from locan.data.region import Region

__all__: list[str] = [
    "distance_to_region",
    "distance_to_region_boundary",
    "max_distance",
    "inertia_moments",
]

logger = logging.getLogger(__name__)


class InertiaMoments(NamedTuple):
    eigenvalues: npt.NDArray[Any]
    eigenvectors: npt.NDArray[Any]
    variance_explained: list[float]
    orientation: float
    eccentricity: float


def distance_to_region(locdata: LocData, region: Region) -> npt.NDArray[np.float_]:
    """
    Determine the distance to the nearest point within `region` for all
    localizations.
    Returns zero if localization is within the region.

    Parameters
    ----------
    locdata
        Localizations for which distances are determined.

    region
        Region from which the closest point is selected.

    Returns
    --------
    npt.NDArray[np.float_]
        Distance for each localization.
    """
    distances = np.full(len(locdata), 0.0)
    if isinstance(region, (Region2D, RoiRegion)):
        for i, point in enumerate(locdata.coordinates):
            distances[i] = Point(point).distance(region.shapely_object)
    else:
        raise NotImplementedError("Region must be Region2D object.")

    return distances


def distance_to_region_boundary(
    locdata: LocData, region: Region
) -> npt.NDArray[np.float_]:
    """
    Determine the distance to the nearest region boundary for all
    localizations.
    Returns a positive value regardless of weather the point is within or
    outside the region.

    Parameters
    ----------
    locdata
        Localizations for which distances are determined.

    region
        Region from which the closest point is selected.

    Returns
    --------
    npt.NDArray[np.float_]
        Distance for each localization.
    """
    distances = np.full(len(locdata), 0.0)
    if isinstance(region, (Region2D, RoiRegion)):
        for i, point in enumerate(locdata.coordinates):
            distances[i] = Point(point).distance(region.shapely_object.boundary)
    else:
        raise NotImplementedError("Region must be Region2D object.")

    return distances


def max_distance(locdata: LocData) -> dict[str, float]:
    """
    Return maximum of all distances between any two localizations in locdata.

    Parameters
    ----------
    locdata
        Localization data

    Returns
    -------
    dict[str, float]
        A dict with key `max_distance` and the corresponding value being the
        maximum distance.
    """
    if not locdata or len(locdata) == 1:
        points = None
    elif len(locdata) == 2:
        points = locdata.coordinates
    elif locdata.convex_hull is None:
        points = None
    else:
        points = locdata.convex_hull.vertices

    distances = np.nan if points is None else pdist(points)
    distance = float(np.nanmax(distances))
    return {"max_distance": distance}


def inertia_moments(points: npt.ArrayLike) -> InertiaMoments:
    """
    Return inertia moments (or principal components) and related properties
    for the given points.
    Inertia moments are represented by eigenvalues (and corresponding
    eigenvectors) of the covariance matrix.
    Variance_explained represents the eigenvalues normalized to the sum of all
    eigenvalues.
    For 2-dimensional data, orientation is the angle (in degrees) between the
    principal axis with the largest variance and the x-axis.
    Also, for 2-dimensional data, eccentricity is computed as
    e=Sqrt(1-M_min/M_max).

    Parameters
    ----------
    points
        Coordinates of input points with shape (npoints, ndim).

    Returns
    -------
    InertiaMoments
        A tuple with eigenvalues, eigenvectors, variance_explained,
        orientation, eccentricity

    Note
    ----
    The data is not standardized.
    """
    points = np.asarray(points)
    covariance_matrix = np.cov(points.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    variance_explained = [
        eigen_value / sum(eigen_values) for eigen_value in eigen_values
    ]
    index_max_eigen_value = np.argmax(eigen_values)

    if np.shape(points)[1] == 2:
        orientation = np.degrees(
            np.arctan2(
                eigen_vectors[1][index_max_eigen_value],
                eigen_vectors[0][index_max_eigen_value],
            )
        )
        eccentricity = np.sqrt(1 - np.min(eigen_values) / np.max(eigen_values))
    else:  # todo implement for 3d
        logger.warning(
            "Orientation and eccentricity have not yet been implemented for 3D."
        )
        orientation = np.nan
        eccentricity = np.nan

    inertia_moments_ = InertiaMoments(
        eigen_values, eigen_vectors, variance_explained, orientation, eccentricity
    )
    return inertia_moments_
