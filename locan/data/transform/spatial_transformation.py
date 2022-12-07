"""

Transform localization data.

This module takes localization data and applies transformation procedures on
coordinates or other properties.

"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from locan.data.locdata import LocData
from locan.data.metadata_utils import _modify_meta
from locan.data.region import Region
from locan.dependencies import HAS_DEPENDENCY
from locan.simulation import simulate_uniform

if HAS_DEPENDENCY["open3d"]:
    import open3d as o3d


__all__ = ["transform_affine", "randomize", "overlay"]

logger = logging.getLogger(__name__)


def _transform_affine_numpy(
    points, matrix=None, offset=None, pre_translation=None
) -> np.ndarray:
    """
    Transform `points` by an affine transformation using standard numpy
    procedures.

    Parameters
    ----------
    points : array-like
        Points on which to perform the manipulation.
    matrix : array-like | None
        Transformation matrix. If None the unit matrix is used.
        Array with shape (ndim, ndim).
        If None the unit matrix is used.
    offset : array-like | None
        Translation vector.
        Array with shape (ndim,).
        If None a vector of zeros is used.
    pre_translation : array-like | None
        Translation vector for coordinates applied before affine
        transformation. Array with shape (ndim,).
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
        # transformed_points = np.array(
        # [np.dot(matrix_, point) + offset_ for point in points_]
        # )
        # same function but better performance:
        transformed_points = np.einsum("ij, nj -> ni", matrix_, points) + offset_
    else:
        transformed_points = points_ + pre_translation
        # transformed_points = np.array(
        # [np.dot(matrix_, point) + offset_ for point in transformed_points]
        # )
        transformed_points = (
            np.einsum("ij, nj -> ni", matrix_, transformed_points) + offset_
        )
        transformed_points = transformed_points - pre_translation

    return transformed_points


def _homogeneous_matrix(matrix=np.identity(3), offset=np.zeros(3)) -> np.ndarray:
    """
    Combine transformation matrix and translation vector for dimension d into
    homogeneous (d+1, d+1) transformation
    matrix to be used with homogeneous coordinates.

    Parameters
    ----------
    matrix : array-like
        Transformation matrix. If None the unit matrix is used.
        Array with shape (ndim, ndim).
    offset : array-like | None
        Translation vector.
        Array with shape (ndim,).

    Returns
    -------
    numpy.ndarray
        Homogeneous transformation matrix to be used with homogeneous
        coordinate vector. Array with shape (ndim+1, ndim+1).
    """
    dimension = np.shape(matrix)[0]

    if dimension != np.shape(matrix)[1]:
        raise ValueError("The matrix has to be of shape (d, d).")

    if dimension != np.shape(offset)[0]:
        raise ValueError("The matrix and offset must have the same dimension.")

    matrix_ = np.identity(dimension + 1)
    matrix_[0:dimension, 0:dimension] = matrix
    matrix_[:dimension, dimension] = offset

    return matrix_


def _transform_affine_open3d(
    points, matrix=None, offset=None, pre_translation=None
) -> np.ndarray:
    """
    Transform `points` or coordinates in `locdata` by an affine
    transformation using open3d.

    Parameters
    ----------
    points : array-like
        Points on which to perform the manipulation.
    matrix : array-like | None
        Transformation matrix. If None the unit matrix is used.
        Array with shape (ndim, ndim).
        If None the unit matrix is used.
    offset : array-like | None
        Translation vector.
        Array with shape (ndim,).
        If None a vector of zeros is used.
    pre_translation : array-like | None
        Translation vector for coordinates applied before affine
        transformation. Array with shape (ndim,).
        The reverse translation is applied after the affine transformation.

    Returns
    -------
    numpy.ndarray
        Transformed coordinates.
    """
    if not HAS_DEPENDENCY["open3d"]:
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
        raise ValueError("Point array has the wrong shape.")

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


def transform_affine(
    locdata, matrix=None, offset=None, pre_translation=None, method="numpy"
) -> np.ndarray | LocData:
    """
    Transform `points` or coordinates in `locdata` by an affine transformation.

    Parameters
    ----------
    locdata : numpy.ndarray | LocData
        Localization data on which to perform the manipulation.
    matrix : array-like | None
        Transformation matrix. If None the unit matrix is used.
        Array with shape (ndim, ndim).
        If None the unit matrix is used.
    offset : array-like | None
        Translation vector.
        Array with shape (ndim,).
        If None a vector of zeros is used.
    pre_translation : array-like | None
        Translation vector for coordinates applied before affine
        transformation. Array with shape (ndim,).
        The reverse translation is applied after the affine transformation.
    method : str
        The method (i.e. library or algorithm) used for computation.
        One of 'numpy', 'open3d'.

    Returns
    -------
    numpy.ndarray | LocData
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

    if method == "numpy":
        transformed_points = _transform_affine_numpy(
            points, matrix=matrix, offset=offset, pre_translation=pre_translation
        )
    elif method == "open3d":
        transformed_points = _transform_affine_open3d(
            points, matrix=matrix, offset=offset, pre_translation=pre_translation
        )
    else:
        raise ValueError(f"Method {method} is not available.")

    # prepare output
    if isinstance(locdata, LocData):
        # new LocData object
        new_dataframe = locdata.data.copy()
        new_dataframe.update(
            pd.DataFrame(
                transformed_points,
                columns=locdata.coordinate_labels,
                index=locdata.data.index,
            )
        )
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

    else:
        return transformed_points


def randomize(locdata, hull_region="bb", seed=None):
    """
    Transform locdata coordinates into randomized coordinates that follow
    complete spatial randomness on the same region as the input locdata.

    Parameters
    ----------
    locdata : LocData
        Localization data to be randomized
    hull_region : Region | str
        Region of interest. String identifier can be one of 'bb', 'ch', 'as',
        'obb' referring to the corresponding hull.
    seed : None, int, array_like[int], numpy.random.SeedSequence, numpy.random.BitGenerator, numpy.random.Generator
        random number generation seed

    Returns
    -------
    locdata : LocData
        New localization data with randomized coordinates.
    """
    local_parameter = locals()

    rng = np.random.default_rng(seed)

    if hull_region == "bb":
        region_ = locdata.bounding_box.hull.T
    elif hull_region == "ch":
        region_ = locdata.convex_hull.region
    elif hull_region == "as":
        region_ = locdata.alpha_shape.region
    elif hull_region == "obb":
        region_ = locdata.oriented_bounding_box.region
    elif isinstance(hull_region, Region):
        region_ = hull_region
    else:
        raise NotImplementedError

    new_locdata = simulate_uniform(n_samples=len(locdata), region=region_, seed=rng)

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


def overlay(locdatas, centers="centroid", orientations=None):
    """
    Translate locdatas to their common center and rotate according to their
    orientation.

    Parameters
    ----------
    locdatas : Iterable[LocData]
        Localization data to overlay.
    centers : Iterable | str | None
        centers to which locdatas are translated.
        Must have the same length as locdatas.
        One of `centroid`, `ch`, 'bb', 'obb', or 'region'.
        If None, no translation is applied.
    orientations : Iterable | str | None
        Orientation value to use in degree.
        Must have the same length as locdatas.
        If str, it must be one of `orientation_im`, `orientation_obb`.
        If None, no rotation is applied.

    Returns
    -------
    LocData
        Collection with transformed locdatas.

    References
    ----------
    .. [1] Broeken J, Johnson H, Lidke DS, et al.
       "Resolution improvement by 3 D particle averaging in localization microscopy"
       Methods and applications in fluorescence 3(1), 014003, 2015,
       doi:10.1088/2050-6120/3/1/014003.
    """
    local_parameter = locals()

    if not locdatas or not isinstance(locdatas, (tuple, list)):
        raise TypeError("locdatas must be a list of LocData objects.")

    dimensions = {locdata.dimension for locdata in locdatas}
    if len(dimensions) == 1:
        dimension = dimensions.pop()
    else:
        raise ValueError(
            "Some elements of `locdatas` have a different dimension. "
            "All locdatas must have the same dimension."
        )

    if centers is None or isinstance(centers, str):
        centers = [centers] * len(locdatas)
    else:
        if len(centers) != len(locdatas):
            raise ValueError("Centers must have the same length as locdatas.")

    if orientations is None or isinstance(orientations, str):
        orientations = [orientations] * len(locdatas)
    else:
        if len(orientations) != len(locdatas):
            raise ValueError("orientations must have the same length as locdatas.")

    transformed_locdatas = []
    for locdata, centre, orientation in zip(locdatas, centers, orientations):
        # translation
        if centre is None:
            transformed_locdata = locdata
        elif not isinstance(centre, str):
            transformed_locdata = transform_affine(
                locdata, matrix=None, offset=np.multiply(centre, -1)
            )
        elif centre == "centroid":
            transformed_locdata = transform_affine(
                locdata, matrix=None, offset=np.multiply(locdata.centroid, -1)
            )
        elif centre == "bb":
            transformed_locdata = transform_affine(
                locdata,
                matrix=None,
                offset=np.multiply(locdata.bounding_box.region.centroid, -1),
            )
        elif centre == "obb":
            transformed_locdata = transform_affine(
                locdata,
                matrix=None,
                offset=np.multiply(locdata.oriented_bounding_box.region.centroid, -1),
            )
        elif centre == "ch":
            transformed_locdata = transform_affine(
                locdata,
                matrix=None,
                offset=np.multiply(locdata.convex_hull.region.centroid, -1),
            )
        elif centre == "region":
            transformed_locdata = transform_affine(
                locdata, matrix=None, offset=np.multiply(locdata.region.centroid, -1)
            )
        else:
            raise ValueError(f"Value for centre={centre} is not defined.")

        # prepare rotation
        if orientation is None:
            matrix = np.identity(dimension)
        elif not isinstance(orientation, str):
            theta = -np.radians(orientation)
            cos_, sin_ = np.cos(theta), np.sin(theta)
            matrix = np.array(((cos_, -sin_), (sin_, cos_)))
        else:
            if dimension != 2:
                raise NotImplementedError(
                    "Rotation has only been implemented for 2 dimensions."
                )
            if orientation == "orientation_obb":
                locdata.oriented_bounding_box
            elif orientation == "orientation_im":
                locdata.inertia_moments
            else:
                raise ValueError(f"orientation={orientation} is undefined.")

            theta = -np.radians(locdata.properties[orientation])
            cos_, sin_ = np.cos(theta), np.sin(theta)
            matrix = np.array(((cos_, -sin_), (sin_, cos_)))

        # rotation
        transformed_locdata = transform_affine(
            transformed_locdata, matrix=matrix, offset=None
        )

        transformed_locdatas.append(transformed_locdata)

    new_locdata = LocData.from_collection(transformed_locdatas)

    # update metadata
    del new_locdata.meta.history[:]
    new_locdata.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(local_parameter)
    )

    return new_locdata
