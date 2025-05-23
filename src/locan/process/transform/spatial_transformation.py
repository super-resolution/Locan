"""

Transform localization data.

This module takes localization data and applies transformation procedures on
coordinates or other properties.

"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.transform import Rotation as spRotation

from locan.data.locdata import LocData
from locan.data.metadata_utils import _modify_meta
from locan.data.validation import _check_loc_properties
from locan.dependencies import HAS_DEPENDENCY, needs_package

if HAS_DEPENDENCY["open3d"]:
    import open3d as o3d


__all__: list[str] = [
    "Rotation2D",
    "Rotation3D",
    "transform_affine",
    "standardize",
    "overlay",
]

logger = logging.getLogger(__name__)


T_Rotation2D = TypeVar("T_Rotation2D", bound="Rotation2D")


class Rotation2D:
    """
    Rotation in 2 dimensions.

    This class provides an interface to initialize from and represent rotations
    with rotation matrices, angles, and unit vectors.

    To create Rotation objects use from_... methods.
    Rotation(...) is not supposed to be instantiated directly.

    This function follows the interface of
    class:`scipy.spatial.transform.Rotation` that can be applied in 3D.

    See also:
        class:`scipy.spatial.transform.Rotation`
        class:`Rotation3D`
    """

    def __init__(self, angle_in_radians: float, rotation_matrix: npt.ArrayLike) -> None:
        self._matrix = np.asarray(rotation_matrix)
        self._angle_in_radians = angle_in_radians

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        return self._matrix

    def as_matrix(self) -> npt.NDArray[np.float64]:
        """
        Return the rotation matrix.

        Returns
        -------
        npt.NDArray[np.float64]
        """
        return self._matrix

    def as_angle(self, degrees: bool = False) -> float:
        """
        Return the rotation angle.

        Parameters
        ----------
        degrees
            If true the angle is given in degrees, else in radians.

        Returns
        -------
        float
        """
        if degrees:
            return float(np.rad2deg(self._angle_in_radians))
        else:
            return self._angle_in_radians

    @classmethod
    def from_matrix(cls: type[T_Rotation2D], matrix: npt.ArrayLike) -> T_Rotation2D:
        """
        Initialize from rotation matrix.

        Parameters
        ----------
        matrix
            Valid 2-dimensional rotation matrix

        Returns
        -------
        Rotation2D
        """
        matrix = np.asarray(matrix)
        if not np.isclose(np.linalg.det(matrix), 1) or not np.allclose(
            matrix @ matrix.T, np.identity(2)
        ):
            raise ValueError("The given matrix is not a valid rotation matrix in 2D.")
        angle = np.arccos(matrix[0, 0])
        return cls(angle_in_radians=angle, rotation_matrix=matrix)

    @classmethod
    def from_angle(
        cls: type[T_Rotation2D], angle: float, degrees: bool = False
    ) -> T_Rotation2D:
        """
        Initialize from rotation angle.

        Parameters
        ----------
        angle
            Angle between x-axis and counter-clockwise rotated axis.
        degrees
            If true, angle is in degrees else in radians.

        Returns
        -------
        Rotation2D
        """
        theta = angle if degrees is False else np.deg2rad(angle)
        cos_, sin_ = np.cos(theta), np.sin(theta)
        matrix = np.array([(cos_, -sin_), (sin_, cos_)])
        return cls(angle_in_radians=theta, rotation_matrix=matrix)

    @classmethod
    def align_vectors(
        cls: type[T_Rotation2D],
        vector: npt.ArrayLike,
        other_vector: npt.ArrayLike = (1, 0),
    ) -> T_Rotation2D:
        """
        Estimate a rotation to optimally align other_vector with vector.

        Parameters
        ----------
        vector
            2-dimensional vector
        other_vector
            2-dimensional vector

        Returns
        -------
        Rotation2D
        """
        vector = np.asarray(vector)
        other_vector = np.asarray(other_vector)
        # dot = x1 * x2 + y1 * y2
        dot = np.vdot(other_vector, vector)
        # det = x1 * y2 - y1 * x2
        det = other_vector[0] * vector[1] - other_vector[1] * vector[0]
        angle = np.atan2(det, dot)
        return cls.from_angle(angle=angle)

    def apply(self, vectors: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Apply this rotation to 2-dimensional vectors.

        Parameters
        ----------
        vectors
            One or more 2-dimensional vectors on which to apply the rotation.

        Returns
        -------
        npt.NDArray[np.float64]
        """
        vectors = np.asarray(vectors)
        return np.matvec(self.matrix, vectors)  # type: ignore[no-any-return]

    def inv(self) -> Rotation2D:
        """
        Provide an instance with the inverted rotation
        (with a rotation matrix that is equal to the transposed matrix).

        Returns
        -------
        Rotation2D
        """
        inverted_matrix = self.matrix.T
        inverted_rotation = Rotation2D.from_matrix(matrix=inverted_matrix)
        return inverted_rotation


class Rotation3D(spRotation):  # type: ignore[misc]
    """
    Rotation in 3 dimensions.

    Adapter class for class:`scipy.spatial.transform.Rotation`.
    """


def _transform_affine_numpy(
    points: npt.ArrayLike,
    matrix: npt.ArrayLike | None = None,
    offset: npt.ArrayLike | None = None,
    pre_translation: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """
    Transform `points` by an affine transformation using standard numpy
    procedures.

    Parameters
    ----------
    points
        Points on which to perform the manipulation.
    matrix
        Transformation matrix. If None the unit matrix is used.
        Array with shape (ndim, ndim).
        If None the unit matrix is used.
    offset
        Translation vector.
        Array with shape (ndim,).
        If None a vector of zeros is used.
    pre_translation
        Translation vector for coordinates applied before affine
        transformation. Array with shape (ndim,).
        The reverse translation is applied after the affine transformation.

    Returns
    -------
    npt.NDArray[np.float64]
        Transformed coordinates.
    """
    points_ = np.asarray(points)
    dimension = np.shape(points_)[1]

    matrix_ = np.identity(dimension) if matrix is None else np.asarray(matrix)
    offset_ = np.zeros(dimension) if offset is None else np.asarray(offset)

    if pre_translation is None:
        # transformed_points = np.array(
        # [np.dot(matrix_, point) + offset_ for point in points_]
        # )
        # same function but better performance:
        transformed_points: npt.NDArray[np.float64] = (
            np.einsum("ij, nj -> ni", matrix_, points_) + offset_
        )
    else:
        pre_translation = np.asarray(pre_translation)
        transformed_points = points_ + pre_translation
        # transformed_points = np.array(
        # [np.dot(matrix_, point) + offset_ for point in transformed_points]
        # )
        transformed_points = (
            np.einsum("ij, nj -> ni", matrix_, transformed_points) + offset_
        )
        transformed_points = transformed_points - pre_translation

    return transformed_points


def _homogeneous_matrix(
    matrix: npt.ArrayLike | None = None, offset: npt.ArrayLike | None = None
) -> npt.NDArray[np.float64]:
    """
    Combine transformation matrix and translation vector for dimension d into
    homogeneous (d+1, d+1) transformation
    matrix to be used with homogeneous coordinates.

    Parameters
    ----------
    matrix
        Transformation matrix. If None the unit matrix is used.
        Array with shape (ndim, ndim).
    offset
        Translation vector.
        Array with shape (ndim,).

    Returns
    -------
    npt.NDArray[np.float64]
        Homogeneous transformation matrix to be used with homogeneous
        coordinate vector. Array with shape (ndim+1, ndim+1).
    """
    matrix = np.identity(3) if matrix is None else np.asarray(matrix)
    offset = np.zeros(3) if offset is None else np.asarray(offset)
    dimension = np.shape(matrix)[0]

    if dimension != np.shape(matrix)[1]:
        raise ValueError("The matrix has to be of shape (d, d).")

    if dimension != np.shape(offset)[0]:
        raise ValueError("The matrix and offset must have the same dimension.")

    matrix_ = np.identity(dimension + 1)
    matrix_[0:dimension, 0:dimension] = matrix
    matrix_[:dimension, dimension] = offset

    return matrix_


@needs_package("open3d")
def _transform_affine_open3d(
    points: npt.ArrayLike,
    matrix: npt.ArrayLike | None = None,
    offset: npt.ArrayLike | None = None,
    pre_translation: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """
    Transform `points` or coordinates in `locdata` by an affine
    transformation using open3d.

    Parameters
    ----------
    points
        Points on which to perform the manipulation.
    matrix
        Transformation matrix.
        Array with shape (ndim, ndim).
        If None the unit matrix is used.
    offset
        Translation vector.
        Array with shape (ndim,).
        If None a vector of zeros is used.
    pre_translation
        Translation vector for coordinates applied before affine
        transformation. Array with shape (ndim,).
        The reverse translation is applied after the affine transformation.

    Returns
    -------
    npt.NDArray[np.float64]
        Transformed coordinates.
    """
    points_ = np.asarray(points)
    dimension = np.shape(points_)[1]

    matrix_ = np.identity(dimension) if matrix is None else np.asarray(matrix)
    offset_ = np.zeros(dimension) if offset is None else np.asarray(offset)

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
    locdata: npt.ArrayLike | LocData,
    matrix: npt.ArrayLike | None = None,
    offset: npt.ArrayLike | None = None,
    pre_translation: npt.ArrayLike | None = None,
    method: Literal["numpy", "open3d"] = "numpy",
) -> npt.NDArray[np.float64] | LocData:
    """
    Transform `points` or coordinates in `locdata` by an affine transformation.

    Parameters
    ----------
    locdata
        Localization data on which to perform the manipulation.
    matrix
        Transformation matrix. If None the unit matrix is used.
        Array with shape (ndim, ndim).
        If None the unit matrix is used.
    offset
        Translation vector.
        Array with shape (ndim,).
        If None a vector of zeros is used.
    pre_translation
        Translation vector for coordinates applied before affine
        transformation. Array with shape (ndim,).
        The reverse translation is applied after the affine transformation.
    method
        The method (i.e. library or algorithm) used for computation.
        One of 'numpy', 'open3d'.

    Returns
    -------
    npt.NDArray[np.float64] | LocData
        New localization data with transformed coordinates.
    """
    local_parameter = locals()

    if len(locdata) == 0:  # type: ignore
        return locdata  # type: ignore

    # adjust input
    if isinstance(locdata, LocData):
        points = locdata.coordinates
    else:
        points = np.asarray(locdata)

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
        df = pd.DataFrame(
            transformed_points,
            columns=locdata.coordinate_keys,
            index=locdata.data.index,
        )
        # cast dtypes
        columns_intersection = list(set(df.columns) & set(locdata.coordinate_keys))
        df = df.astype(new_dataframe[columns_intersection].dtypes)

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

    else:
        return transformed_points


def standardize(
    locdata: LocData,
    loc_properties: list[str] | None = None,
    with_mean: bool = True,
    with_std: bool = True,
) -> LocData:
    """
    Transform locdata properties by centering to the mean
    and property-wise scaling to unit standard deviation (variance).

    Note
    -----
    This function makes use of :func:sklearn.preprocessing.scale
    and thus works with a biased estimator for the standard deviation.

    Parameters
    ----------
    locdata
        Localization data to be standardized.
    loc_properties
        Localization properties to be standardized.
        If None The coordinate_values of locdata are used.
    with_mean
        If True center to the mean.
    with_std
        If True scale to unit standard deviation (variance).

    Returns
    -------
    LocData
        New localization data with standardized properties.
    """
    local_parameter = locals()

    if len(locdata) == 0:
        return locdata

    labels_ = _check_loc_properties(locdata, loc_properties)
    data = locdata.data[labels_].values

    from sklearn.preprocessing import scale

    transformed_data = scale(data, with_mean=with_mean, with_std=with_std)

    new_dataframe = locdata.data.copy()
    df = pd.DataFrame(
        transformed_data,
        columns=labels_,
        index=locdata.data.index,
    )
    # cast dtypes
    columns_intersection = list(set(df.columns) & set(labels_))
    df = df.astype(new_dataframe[columns_intersection].dtypes)
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


def overlay(
    locdatas: Iterable[LocData],
    centers: Iterable[Any] | str | None = "centroid",
    orientations: Sequence[int | float | str | None] | str | None = None,
) -> LocData:
    """
    Translate locdatas to their common center and rotate according to their
    orientation.

    Parameters
    ----------
    locdatas
        Localization data to overlay.
    centers
        centers to which locdatas are translated.
        Must have the same length as locdatas.
        One of `centroid`, `ch`, 'bb', 'obb', or 'region'.
        If None, no translation is applied.
    orientations
        Orientation value to use in degree.
        Must have the same length as locdatas.
        If str, it must be one of `orientation_im`, `orientation_obb`.
        If None, no rotation is applied.

    Returns
    -------
    LocData
        Collection with transformed locdatas.
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
        if len(centers) != len(locdatas):  # type: ignore
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
                locdata.oriented_bounding_box  # update oriented_bounding_box  # noqa B018
            elif orientation == "orientation_im":
                locdata.inertia_moments  # update inertia_moments  # noqa B018
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
