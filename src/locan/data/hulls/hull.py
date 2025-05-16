"""

Hull objects of localization data.

This module computes specific hulls for the bounding box, convex hull and
oriented bounding box and related properties for LocData objects.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.spatial as spat
from shapely.geometry import MultiPoint as shMultiPoint
from shapely.geometry import Polygon as shPolygon

from locan.data.adapter.adapter_open3d import points_to_open3d
from locan.data.regions.region import (
    AxisOrientedCuboid,
    AxisOrientedRectangle,
    Cuboid,
    EmptyRegion,
    Polygon,
    Rectangle,
)
from locan.dependencies import HAS_DEPENDENCY, needs_package

if HAS_DEPENDENCY["open3d"]:
    import open3d as o3d

if TYPE_CHECKING:
    from locan.data.regions.region import Region

__all__: list[str] = ["BoundingBox", "ConvexHull", "OrientedBoundingBox"]


class BoundingBox:
    """
    Class with bounding box computed using numpy operations.

    Parameters
    ----------
    points : npt.ArrayLike
        Coordinates of input points with shape (npoints, ndim).

    Attributes
    ----------
    hull : npt.NDArray[np.float64]
        Array of point coordinates of shape (2, ndim) that represent
        [[min_coordinates], [max_coordinates]].
    dimension : int
        Spatial dimension of hull
    vertices : npt.NDArray[np.float64]
        Coordinates of points that make up the hull.
        Array of shape (ndim, 2).
    width : npt.NDArray[np.float64]
        Array with differences between max and min for each coordinate.
    region_measure : float
        Hull measure, i.e. area or volume
    subregion_measure : float
        Measure of the sub-dimensional region, i.e. circumference or surface
    region : Region
        Convert the hull to a Region object.
    """

    def __init__(self, points: npt.ArrayLike) -> None:
        points = np.asarray(points)
        if np.size(points) == 0:
            self.dimension: int = 0
            self.hull = np.array([])
            self.width = np.zeros(self.dimension)
            self.region_measure: float = 0
            self.subregion_measure: float = 0
        elif len(points) < 2:
            self.dimension = np.shape(points)[1]
            self.hull = np.array([])
            self.width = np.zeros(self.dimension)
            self.region_measure = 0
            self.subregion_measure = 0
        else:
            self.dimension = np.shape(points)[1]
            self.hull = np.array([np.min(points, axis=0), np.max(points, axis=0)])
            self.width = np.diff(self.hull, axis=0).flatten()
            self.region_measure = np.prod(self.width)  # type: ignore
            self.subregion_measure = np.sum(self.width) * 2

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        return self.hull.T

    @property
    def region(self) -> Region:
        region_: Region
        if self.dimension == 2:
            region_ = AxisOrientedRectangle(self.hull[0], self.width[0], self.width[1])
        elif self.dimension == 3:
            region_ = AxisOrientedCuboid(
                self.hull[0],
                length=self.width[0],
                width=self.width[1],
                height=self.width[2],
            )
        # todo: implement higher dimensions
        else:
            raise NotImplementedError
        return region_


class _ConvexHullScipy:
    """
    Class with convex hull computed using the scipy.spatial.ConvexHull method.

    Parameters
    ----------
    points : npt.ArrayLike
        Coordinates of input points. Array with shape (npoints, ndim).

    Attributes
    ----------
    hull : scipy.spatial.ConvexHull
        hull object from the corresponding algorithm
    dimension : int
        spatial dimension of hull
    vertices : npt.NDArray[np.float64]
        Coordinates of points that make up the hull.
        Array of shape (ndim, 2).
    vertex_indices : npt.NDArray[np.int64]
        Indices identifying a polygon of all points that make up the hull.
    points_on_boundary : int
        absolute number of points that are part of the convex hull.
    points_on_boundary_rel : float
        The number of points on the hull relative to all input points
    region_measure : float
        hull measure, i.e. area or volume
    subregion_measure : float
        measure of the sub-dimensional region, i.e. circumference or surface
    region : Region
        Convert the hull to a Region object.
    """

    def __init__(self, points: npt.ArrayLike) -> None:
        points = np.asarray(points)
        if len(points) < 6:
            unique_points = np.array(list(set(tuple(point) for point in points)))
            if len(unique_points) < 3:
                raise TypeError(
                    "Convex_hull needs at least 3 different points as input."
                )

        self.dimension = np.shape(points)[1]
        self.hull = spat.ConvexHull(points)
        self.vertex_indices = self.hull.vertices
        self.points_on_boundary = len(self.vertex_indices)
        self.points_on_boundary_rel = self.points_on_boundary / len(points)
        self.region_measure = self.hull.volume
        self.subregion_measure = self.hull.area

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        return_value: npt.NDArray[np.float64] = self.hull.points[self.hull.vertices]
        return return_value

    @property
    def region(self) -> Polygon:
        if self.dimension == 1:
            raise NotImplementedError
        elif self.dimension == 2:
            # closed_vertices = np.append(self.vertices, [self.vertices[0]], axis=0)
            # region_ = RoiRegion(region_type='polygon', region_specs=closed_vertices)
            return Polygon(self.vertices)
        else:
            raise NotImplementedError


class _ConvexHullShapely:
    """
    Class with convex hull computed using the scipy.spatial.ConvexHull method.

    Parameters
    ----------
    points : npt.ArrayLike
        Coordinates of input points with shape (npoints, ndim).

    Attributes
    ----------
    hull : Hull
        Polygon object from the .convex_hull method
    dimension : int
        Spatial dimension of hull
    vertices : npt.NDArray[np.float64]
        Coordinates of points that make up the hull.
        Array of shape (ndim, 2).
    vertex_indices : npt.NDArray[np.int64]
        indices identifying a polygon of all points that make up the hull
    points_on_boundary : int
        The absolute number of points on the hull
    points_on_boundary_rel : int
        The number of points on the hull relative to all input points
    region_measure : float
        hull measure, i.e. area or volume
    subregion_measure : float
        measure of the sub-dimensional region, i.e. circumference or surface
    region : Region
        Convert the hull to a RoiRegion object.
    """

    def __init__(self, points: npt.ArrayLike) -> None:
        points = np.asarray(points)
        if len(points) < 6:
            unique_points = np.array(list(set(tuple(point) for point in points)))
            if len(unique_points) < 3:
                raise TypeError(
                    "Convex_hull needs at least 3 different points as input."
                )

        self.dimension = np.shape(points)[1]
        if self.dimension != 2:
            raise TypeError(
                "ConvexHullShapely only takes 2-dimensional points as input."
            )

        self.hull = shMultiPoint(points).convex_hull
        # todo: set vertex_indices
        # self.vertex_indices = None
        self.points_on_boundary = (
            len(self.hull.exterior.coords) - 1  # type: ignore
        )  # the first point is repeated in exterior.coords
        self.points_on_boundary_rel = self.points_on_boundary / len(points)
        self.region_measure = self.hull.area
        self.subregion_measure = self.hull.length

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        return np.array(self.hull.exterior.coords)[:-1]  # type: ignore

    @property
    def region(self) -> Polygon:
        if self.dimension > 2:
            raise NotImplementedError(
                "Region for 3D data has not yet been implemented."
            )
        else:
            #  closed_vertices = np.append(self.vertices, [self.vertices[0]], axis=0)
            # region_ = RoiRegion(region_type='polygon', region_specs=closed_vertices)
            return Polygon(self.vertices)


class ConvexHull:
    """
    Class with convex hull of localization data.

    Parameters
    ----------
    points : npt.ArrayLike
        Coordinates of input points. Array with shape (npoints, ndim).
    method : Literal['scipy', 'shapely']
        Specific class to compute the convex hull and attributes.
        One of 'scipy', 'shapely'.

    Attributes
    ----------
    method : Literal['scipy', 'shapely']
        Specific class to compute the convex hull and attributes.
        One of 'scipy', 'shapely'.
    hull : Hull
        Polygon object from the .convex_hull method
    dimension : int
        Spatial dimension of hull
    vertices : npt.NDArray[np.float64]
        Coordinates of points that make up the hull.
        Array of shape (ndim, 2).
    vertex_indices : npt.NDArray[np.int64]
        indices identifying a polygon of all points that make up the hull
    points_on_boundary : int
        The absolute number of points on the hull
    points_on_boundary_rel : int
        The number of points on the hull relative to all input points
    region_measure : float
        hull measure, i.e. area or volume
    subregion_measure : float
        measure of the sub-dimensional region, i.e. circumference or surface
    region : Region
        Convert the hull to a Region object.
    """

    def __init__(
        self, points: npt.ArrayLike, method: Literal["scipy", "shapely"] = "scipy"
    ) -> None:
        points = np.asarray(points)
        self._special_class: _ConvexHullScipy | _ConvexHullShapely | None = None

        if np.size(points) == 0:
            self.dimension: int = 0
            self.hull = None
            self.vertices = np.array([])
            self.vertex_indices = np.array([])
            self.points_on_boundary = 0
            self.points_on_boundary_rel = 0
            self.region_measure: float = 0
            self.subregion_measure: float = 0

        elif np.shape(points)[0] == 1:
            self.dimension = np.shape(points)[1]
            self.hull = None
            self.vertices = np.array([])
            self.vertex_indices = np.array([])
            self.points_on_boundary = 0
            self.points_on_boundary_rel = 0
            self.region_measure = 0
            self.subregion_measure = 0

        else:
            self.method = method
            if method == "scipy":
                self._special_class = _ConvexHullScipy(points)
            elif method == "shapely":
                self._special_class = _ConvexHullShapely(points)
            else:
                raise ValueError(f"The provided method {method} is not available.")

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError

        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._special_class, attr)


class _OrientedBoundingBoxShapely:
    """
    Class with oriented bounding box computed using the shapely
    minimum_rotated_rectangle method.

    Parameters
    ----------
    points : npt.ArrayLike
        Coordinates of input points with shape (npoints, ndim).

    Attributes
    ----------
    hull : Polygon
        Polygon object from the minimum_rotated_rectangle method
    dimension : int
        Spatial dimension of hull
    vertices : npt.NDArray[np.float64]
        Coordinates of points that make up the hull.
        Array of shape (ndim, 2).
    width : npt.NDArray[np.float64]
        Array with lengths of box edges.
    region_measure : float
        hull measure, i.e. area or volume
    subregion_measure : float
        measure of the sub-dimensional region, i.e. circumference or surface
    region : Region
        Convert the hull to a Region object.
    angle : float
        Orientation defined as angle (in degrees) between the vector from
        first to last point and x-axis.

    """

    def __init__(self, points: npt.ArrayLike) -> None:
        points = np.asarray(points)
        self.dimension = np.shape(points)[1]

        if len(points) < 6:
            unique_points = np.array(list(set(tuple(point) for point in points)))
            if len(unique_points) < 3:
                raise TypeError(
                    "OrientedBoundingBox needs at least 3 different points as input."
                )

        if self.dimension != 2:
            raise TypeError(
                "OrientedBoundingBox only takes 2-dimensional points as input."
            )

        if len(points) < 3:
            self.hull: npt.NDArray[np.float64] = np.array([])
            self.width = np.zeros(self.dimension)
            self.region_measure = 0
            self.subregion_measure = 0
            self.angle = np.nan
            self.elongation = np.nan
        else:
            self.hull: shPolygon = shMultiPoint(points).minimum_rotated_rectangle.normalize()  # type: ignore
            difference = np.diff(self.vertices[0:3], axis=0)
            # vertices are counter-clockwise listed. To get width in (x, y)
            # we have to look at difference[1], difference[0], respectively:
            self.width = np.array(
                [np.linalg.norm(difference[1]), np.linalg.norm(difference[0])]
            )
            self.region_measure = self.hull.area  # type: ignore[attr-defined]
            self.subregion_measure = self.hull.length  # type: ignore[attr-defined]
            self.angle = float(
                np.degrees(np.arctan2(difference[1][1], difference[1][0]))
            )
            # numpy.arctan2(y, x) takes reversed x, y arguments.
            self.elongation = 1 - np.divide(*sorted(self.width))  # type: ignore

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        return np.array(self.hull.exterior.coords)  # type: ignore[attr-defined]

    @property
    def region(self) -> Rectangle:
        if self.dimension == 2:
            return Rectangle(self.vertices[0], self.width[0], self.width[1], self.angle)
        else:
            raise NotImplementedError


@needs_package("open3d")
class _OrientedBoundingBoxOpen3D:
    """
    Class with minimal oriented bounding box computed using open3D.
    The bounding box is oriented such that its volume is minimized.

    Parameters
    ----------
    points : npt.ArrayLike
        Coordinates of input points with shape (npoints, ndim).

    Attributes
    ----------
    hull : open3d.t.geometry.OrientedBoundingBox
        OrientedBoundingBox object from the
        PointCloud.get_minimal_oriented_bounding_box() method
    dimension : int
        Spatial dimension of hull
    vertices : npt.NDArray[np.float64]
        Coordinates of points that make up the hull.
        Array of shape (ndim, 2).
    width : npt.NDArray[np.float64]
        Array with lengths of box edges.
    region_measure : float
        hull measure, i.e. area or volume
    subregion_measure : float
        measure of the sub-dimensional region, i.e. circumference or surface
    region : Region
        Convert the hull to a Region object.
    """

    def __init__(self, points: npt.ArrayLike) -> None:
        points = np.asarray(points)
        self.dimension = np.shape(points)[1]
        self.hull: npt.NDArray[np.float64] | o3d.t.geometry.OrientedBoundingBox

        if len(points) < 6:
            unique_points = np.array(list(set(tuple(point) for point in points)))
            if len(unique_points) < 3:
                raise TypeError(
                    "OrientedBoundingBox needs at least 3 different points as input."
                )

        point_cloud_open3d = points_to_open3d(points=points)

        if self.dimension == 3:
            if len(points) < 3:
                self.hull = np.array([])
                self.width = np.zeros(self.dimension)  # type: ignore
                self.region_measure = 0
                self.subregion_measure = 0
                self.elongation = np.nan
            else:
                self.hull = point_cloud_open3d.get_oriented_bounding_box()
                self.width = self.hull.extent.numpy()
                self.subregion_measure = 2 * (
                    self.width[0] * self.width[1]
                    + self.width[1] * self.width[2]
                    + self.width[2] * self.width[0]
                )
                self.region_measure = self.hull.volume()
                self.elongation = 1 - np.divide(  # type: ignore[call-overload]
                    *[sorted(self.width)[i] for i in [0, 2]]  # type: ignore[type-var]
                )
        else:
            raise TypeError(
                "_OrientedBoundingBoxOpen3d only takes 3-dimensional points as input."
            )

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        if not self.hull:
            return np.array([])
        else:
            return np.array(self.hull.get_box_points())  # type: ignore[union-attr]

    @property
    def region(self) -> Cuboid | EmptyRegion:
        if self.dimension == 3:
            return Cuboid.from_open3d(self.hull)
        else:
            raise NotImplementedError


class OrientedBoundingBox:
    """
    Class with minimal oriented bounding box of localization data.

    Parameters
    ----------
    points
        Coordinates of input points with shape (npoints, ndim).
    method
        Specific class to compute the convex hull and attributes.

    Attributes
    ----------
    method : Literal['shapely', 'open3d'] | None
        Specific class to compute the convex hull and attributes.
    hull : Polygon
        Object from the minimum_rotated_rectangle method
    dimension : int
        Spatial dimension of hull
    vertices : npt.NDArray[np.float64]
        Coordinates of points that make up the hull.
        Array of shape (ndim, 2).
    width : npt.NDArray[np.float64]
        Array with lengths of box edges.
    region_measure : float
        hull measure, i.e. area or volume
    subregion_measure : float
        measure of the sub-dimensional region, i.e. circumference or surface
    region : Region
        Convert the hull to a Region object.
    """

    def __init__(
        self, points: npt.ArrayLike, method: Literal["shapely", "open3d"] | None = None
    ) -> None:
        points = np.asarray(points)
        self.dimension = np.shape(points)[1]
        self._special_class: (
            _OrientedBoundingBoxShapely | _OrientedBoundingBoxOpen3D | None
        ) = None

        if np.size(points) == 0 or len(points) < 3:
            raise TypeError(
                "OrientedBoundingBox needs at least 3 different points as input."
            )

        if method is None and self.dimension == 2:
            self._special_class = _OrientedBoundingBoxShapely(points)
        elif method is None and self.dimension == 3:
            self._special_class = _OrientedBoundingBoxOpen3D(points)
        elif method is None:
            raise TypeError(
                "OrientedBoundingBox only takes 2- or3-dimensional points as input."
            )
        elif method == "shapely":
            self._special_class = _OrientedBoundingBoxShapely(points)
        elif method == "open3d":
            self._special_class = _OrientedBoundingBoxOpen3D(points)
        else:
            raise ValueError(f"The provided method {method} is not available.")

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError

        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._special_class, attr)
