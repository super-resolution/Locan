"""

Rotation

This module provides helper classes to represent vector rotation.

The class definitions follow the interface of
class:`scipy.spatial.transform.Rotation`.

"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy import typing as npt
from scipy.spatial.transform import Rotation as spRotation

__all__: list[str] = [
    "Rotation2D",
    "Rotation3D",
]


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
        cos_, sin_ = matrix[0, 0], (-1) * matrix[0, 1]
        angle = np.atan2(sin_, cos_)
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
