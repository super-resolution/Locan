"""

Regions as support for localization data.

This module provides classes to define geometric regions for localization data.
All region classes inherit from the abstract base class `Region`.

"""
# todo: fix docstrings
from __future__ import annotations

import itertools as it
import sys
from abc import ABC, abstractmethod
from typing import Any, TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import matplotlib as mpl
import matplotlib.patches as mpl_patches
import matplotlib.path as mpl_path
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist
from shapely.affinity import rotate, scale, translate
from shapely.geometry import MultiPoint as shMultiPoint
from shapely.geometry import MultiPolygon as shMultiPolygon
from shapely.geometry import Point as shPoint
from shapely.geometry import Polygon as shPolygon
from shapely.prepared import prep

__all__: list[str] = [
    "Region",
    "EmptyRegion",
    "Region1D",
    "Interval",
    "Region2D",
    "Rectangle",
    "Ellipse",
    "Polygon",
    "MultiPolygon",
    "Region3D",
    "AxisOrientedCuboid",
    "Cuboid",
    "RegionND",
    "AxisOrientedHypercuboid",
]

# __all__ += ['Ellipsoid' 'Polyhedron']
# __all__ += ['Polytope']

__all__ += ["RoiRegion"]  # legacy code that is only needed for legacy _roi.yml files.


class RoiRegion:
    """
    Deprecated Region object to specify regions of interest.

    A region that defines a region of interest with methods for getting a
    printable representation (that can also be saved in a yaml file),
    for returning a matplotlib patch that can be shown in a graph,
    for finding points within the region.

    Warnings
    ________
    This class is to be deprecated and should only be used to deal with
    legacy _roi.yaml files.
    Use Region classes instead.

    Parameters
    ----------
    region_type : str
        A string indicating the roi shape.
        In 1D it can be `interval`.
        In 2D it can be either `rectangle`, `ellipse`, or closed `polygon`.
        In 2D it can also be `shapelyPolygon` or `shapelyMultiPolygon`.
        In 3D it can be either `cuboid` or `ellipsoid` or `polyhedron`
        (not implemented yet).
    region_specs : tuple
        1D rois are defined by the following tuple:
        * interval: (start, stop)
        2D rois are defined by the following tuples:
        * rectangle: ((corner_x, corner_y), width, height, angle)
        with angle in degree
        * ellipse: ((center_x, center_y), width, height, angle)
        with angle in degree
        * polygon: ((point1_x, point1_y), (point2_x, point2_y), ...,
        (point1_x, point1_y))
        * shapelyPolygon: ((point_tuples), ((hole_tuples), ...))
        * shapelyMultiPolygon: (shapelyPolygon_specs_1, shapelyPolygon_specs_2,
        ...)
        3D rois are defined by the following tuples:
        * cuboid: ((corner_x, corner_y, corner_z), length, width, height,
        angle_1, angle_2, angle_3)
        * ellipsoid: ((center_x, center_y, center_z), length, width, height,
        angle_1, angle_2, angle_3)
        * polyhedron: (...)

    Attributes
    ----------
    region_type : str
        Type of region
    region_specs : tuple
        Specifications for region
    _region : RoiRegion
        RoiRegion instance for the specified region type.
    polygon : tuple[npt.ArrayLike, ...]
        Array of points for a closed polygon approximating the region of
        interest in clockwise orientation.
        The first and last point must be identical.
    dimension : int
        Spatial dimension of region
    centroid : tuple[float, ...]
        Centroid coordinates
    max_distance : npt.NDArray[np.float_]
        Maximum distance between any two points in the region
    region_measure : float
        Hull measure, i.e. area or volume
    subregion_measure : float
        Measure of the sub-dimensional region, i.e. circumference or surface.
    """

    def __init__(self, region_type: str, region_specs: tuple[Any, ...]) -> None:
        self.region_specs = region_specs
        self.region_type = region_type
        self._region: Region

        if region_type == "interval":
            self._region = Interval(*region_specs)

        elif region_type == "rectangle":
            self._region = Rectangle(*region_specs)

        elif region_type == "ellipse":
            self._region = Ellipse(*region_specs)

        elif region_type == "polygon":
            self._region = Polygon(region_specs)

        elif region_type == "shapelyPolygon":
            self._region = Polygon(region_specs)

        elif region_type == "shapelyMultiPolygon":
            self._region = MultiPolygon(region_specs)  # type: ignore

        else:
            raise NotImplementedError(
                f"Region_type {region_type} has not been implemented yet."
            )

    def __getattr__(self, attr):  # type: ignore
        """All non-adapted calls are passed to the _region object"""
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self._region, attr)

    def __repr__(self):  # type: ignore
        """
        Readable, printable and savable representation of RoiRegion.
        """
        return str(dict(region_type=self.region_type, region_specs=self.region_specs))

    @property
    def region(self):  # type: ignore
        return self._region

    @classmethod
    def from_shapely(cls, region_type, shapely_obj):  # type: ignore
        if region_type == "shapelyPolygon":
            region_specs = Polygon.from_shapely(shapely_obj).region_specs  # type: ignore

        elif region_type == "shapelyMultiPolygon":
            region_specs = MultiPolygon.from_shapely(shapely_obj).region_specs

        else:
            raise NotImplementedError(
                f"Region_type {region_type} has not been implemented yet."
            )

        return cls(region_type=region_type, region_specs=region_specs)

    def contains(self, points) -> npt.NDArray[np.int_]:  # type: ignore
        """
        Return list of indices for all points that are inside the region of
        interest.

        Parameters
        ----------
        points : npt.ArrayLike
            2D or 3D coordinates of oints that are tested for being inside the
            specified region.

        Returns
        -------
        npt.NDArray[np.int_]
            Array with indices for all points in original point array that are
            within the region.
        """
        return self._region.contains(points)  # type: ignore

    def as_artist(self, **kwargs: Any) -> mpl_patches.Patch:
        """
        Matplotlib patch object for this region
        (e.g. `matplotlib.patches.Ellipse`).

        Parameters
        ----------
        kwargs
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        matplotlib.patches.Patch
            Matplotlib patch for the specified region.
        """
        return self._region.as_artist(**kwargs)  # type: ignore

    def to_shapely(self):  # type: ignore
        """
        Convert region to a polygon and return as shapely object.

        Returns
        -------
        shapely.Polygon
        """
        return self._region.shapely_object  # type: ignore


class Region(ABC):
    """
    Abstract Region class to define the interface for Region-derived classes
    that specify geometric objects to represent regions of interest.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(...)"

    @staticmethod
    def from_intervals(
        intervals: npt.ArrayLike,
    ) -> Interval | Rectangle | AxisOrientedCuboid | AxisOrientedHypercuboid:
        """
        Constructor for instantiating axis-oriented, box-like Region from list of
        (min, max) bounds.
        Takes array-like intervals instead of interval to be consistent with
        `Rectangle.from_intervals`.

        Parameters
        ----------
        intervals
            The region bounds for each dimension of shape (dimension, 2).

        Returns
        -------
        Interval | Rectangle | AxisOrientedCuboid | AxisOrientedHypercuboid
        """
        if np.shape(intervals) == (2,):
            return Interval.from_intervals(intervals)
        elif np.shape(intervals) == (2, 2):
            return Rectangle.from_intervals(intervals)
        elif np.shape(intervals) == (3, 2):
            return AxisOrientedCuboid.from_intervals(intervals)
        elif np.shape(intervals)[0] > 3 and np.shape(intervals)[1] == 2:
            return AxisOrientedHypercuboid.from_intervals(intervals)
        else:
            raise TypeError("intervals must be of shape (dimension, 2).")

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        """
        The region dimension.

        Returns
        -------
        int | None
        """
        pass

    @property
    @abstractmethod
    def bounds(self) -> npt.NDArray[np.float_] | None:
        """
        Region bounds min_x, min_y, ..., max_x, max_y, ... for each dimension.

        Returns
        -------
        npt.NDArray[np.float_] | None
            of shape (2 * dimension,)
        """
        pass

    @property
    @abstractmethod
    def extent(self) -> npt.NDArray[np.float_] | None:
        """
        The extent (max_x - min_x), (max_y - min_y), ... for each dimension.

        Returns
        -------
        npt.NDArray[np.float_] | None
            of shape (dimension,)
        """
        pass

    @property
    @abstractmethod
    def points(self) -> npt.NDArray[np.float_] | list[npt.NDArray[np.float_]]:
        """
        Point coordinates.

        Returns
        -------
        npt.NDArray[np.float_] | list[npt.NDArray[np.float_]]
            of shape (n_points, dimension)
        """
        pass

    @property
    @abstractmethod
    def centroid(self) -> npt.NDArray[np.float_] | None:
        """
        Point coordinates for region centroid.

        Returns
        -------
        npt.NDArray[np.float_] | None
            of shape (dimension,)
        """
        pass

    @property
    @abstractmethod
    def max_distance(self) -> float:
        """
        The maximum distance between any two points within the region.

        Returns
        -------
        float
        """
        pass

    @property
    @abstractmethod
    def region_measure(self) -> float:
        """
        Region measure, i.e. area (for 2d) or volume (for 3d).

        Returns
        -------
        float
        """
        pass

    @property
    @abstractmethod
    def subregion_measure(self) -> float:
        """
        Measure of the sub-dimensional region, i.e. circumference (for 2d)
        or surface (for 3d).

        Returns
        -------
        float
        """
        pass

    @property
    @abstractmethod
    def bounding_box(self) -> Region:
        """
        A region describing the minimum axis-aligned bounding box that
        encloses the original region.

        Returns
        -------
        Region
        """
        pass

    @abstractmethod
    def intersection(self, other: Region) -> Region:
        """
        Returns a region representing the intersection of this region with
        ``other``.

        Parameters
        ----------
        other : Region
            Other region

        Returns
        -------
        Region
        """
        raise NotImplementedError

    @abstractmethod
    def symmetric_difference(self, other: Region) -> Region:
        """
        Returns the union of the two regions minus any areas contained in the
        intersection of the two regions.

        Parameters
        ----------
        other : Region
            Other region

        Returns
        -------
        Region
        """
        raise NotImplementedError

    @abstractmethod
    def union(self, other: Region) -> Region:
        """
        Returns a region representing the union of this region with ``other``.

        Parameters
        ----------
        other : Region
            Other region

        Returns
        -------
        Region
        """
        raise NotImplementedError

    def __and__(self, other: Region) -> Region:
        return self.intersection(other)

    def __or__(self, other: Region) -> Region:
        return self.union(other)

    def __xor__(self, other: Region) -> Region:
        return self.symmetric_difference(other)

    @abstractmethod
    def contains(self, points: npt.ArrayLike) -> npt.NDArray[np.int_ | np.int64]:
        """
        Return list of indices for all points that are inside the region
        of interest.

        Parameters
        ----------
        points
            Coordinates of points that are tested for being inside the
            specified region.

        Returns
        -------
        npt.NDArray[np.int_]
            Array with indices for all points in original point array that are
            within the region.
        """
        pass

    def __contains__(self, item: Any) -> bool:
        return True if list(self.contains([item])) == [0] else False

    @abstractmethod
    def buffer(self, distance: float, **kwargs: Any) -> Region:
        """
        Extend the region perpendicular by a `distance`.

        Parameters
        ----------
        distance
            Distance by which the region is extended.

        Returns
        -------
        Region
            The extended region.
        """
        pass


class Region1D(Region):
    """
    Abstract Region class to define the interface for 1-dimensional Region classes.
    """

    @property
    def dimension(self) -> int:
        return 1

    @abstractmethod
    def as_artist(
        self, origin: npt.ArrayLike = (0, 0), **kwargs: Any
    ) -> mpl_patches.Patch:
        """
        Matplotlib 2D patch object for this region
        (e.g. `matplotlib.patches.Ellipse`).

        Parameters
        ----------
        origin
            The (x, y) pixel position of the origin of the displayed image.
            Default is (0, 0).
        kwargs
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        matplotlib.patches.Patch
            Matplotlib patch for the specified region.
        """
        pass

    def intersection(self, other: Region) -> Region:
        raise NotImplementedError

    def symmetric_difference(self, other: Region) -> Region:
        raise NotImplementedError

    def union(self, other: Region) -> Region:
        raise NotImplementedError


class Region2D(Region):
    """
    Abstract Region class to define the interface for 2-dimensional
    Region classes.
    """

    def __getstate__(self) -> Any:
        state = self.__dict__.copy()
        del state["_shapely_object"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__dict__["_shapely_object"] = None

    @property
    def dimension(self) -> int:
        return 2

    @property
    def bounds(self) -> npt.NDArray[np.float_]:
        min_x, min_y, max_x, max_y = self.shapely_object.bounds
        return np.array([min_x, min_y, max_x, max_y])

    @property
    def extent(self) -> npt.NDArray[np.float_]:
        min_x, min_y, max_x, max_y = self.bounds
        return np.array([abs(max_x - min_x), abs(max_y - min_y)])

    @property
    def bounding_box(self) -> Rectangle:
        min_x, min_y, max_x, max_y = self.bounds
        return Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 0)

    @property
    @abstractmethod
    def shapely_object(self) -> Any:
        """
        Geometric object as defined in `shapely`.

        Returns
        -------
        Shapely object
        """
        pass

    @staticmethod
    def from_shapely(
        shapely_object: shPolygon | shMultiPolygon,
    ) -> Polygon | MultiPolygon | EmptyRegion:
        """
        Constructor for instantiating Region from `shapely` object.

        Parameters
        ----------
        shapely_object
            Geometric object to be converted into Region

        Returns
        -------
        Polygon | MultiPolygon | EmptyRegion
        """
        ptype = shapely_object.geom_type
        if ptype == "Polygon":
            return Polygon.from_shapely(shapely_object)
        elif ptype == "MultiPolygon":
            return MultiPolygon.from_shapely(shapely_object)
        else:
            raise TypeError(f"shapely_object cannot be of type {ptype}")

    @abstractmethod
    def as_artist(
        self, origin: npt.ArrayLike = (0, 0), **kwargs: Any
    ) -> mpl_patches.Patch:
        """
        Matplotlib 2D patch object for this region
        (e.g. `matplotlib.patches.Ellipse`).

        Parameters
        ----------
        origin
            The (x, y) pixel position of the origin of the displayed image.
            Default is (0, 0).
        kwargs
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        matplotlib.patches.Patch
            Matplotlib patch for the specified region.
        """
        pass

    def plot(self, ax: mpl.axes.Axes | None = None, **kwargs: Any) -> mpl.axes.Axes:
        """
        Provide plot of region as :class:`matplotlib.axes.Axes` object.

        Parameters
        ----------
        ax
            The axes on which to show the image
        kwargs
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        artist = self.as_artist(**kwargs)
        ax.add_artist(artist)

        return ax

    def intersection(self, other: Region) -> Polygon | MultiPolygon | EmptyRegion:
        if not isinstance(other, (Region2D, EmptyRegion, RoiRegion)):
            raise TypeError("other must be of type Region2D")
        shapely_obj = self.shapely_object.intersection(other.shapely_object)
        return Region2D.from_shapely(shapely_obj)

    def symmetric_difference(
        self, other: Region
    ) -> Polygon | MultiPolygon | EmptyRegion:
        if not isinstance(other, (Region2D, EmptyRegion, RoiRegion)):
            raise TypeError("other must be of type Region2D")
        shapely_obj = self.shapely_object.symmetric_difference(other.shapely_object)
        return Region2D.from_shapely(shapely_obj)

    def union(self, other: Region) -> Polygon | MultiPolygon | EmptyRegion:
        if not isinstance(other, (Region2D, EmptyRegion, RoiRegion)):
            raise TypeError("other must be of type Region2D")
        shapely_obj = self.shapely_object.union(other.shapely_object)
        return Region2D.from_shapely(shapely_obj)

    def buffer(
        self, distance: float, **kwargs: Any
    ) -> Polygon | MultiPolygon | EmptyRegion:
        """
        Extend the region perpendicular by a `distance`.

        Parameters
        ----------
        distance
            Distance by which the region is extended.
        kwargs
            Other parameters passed to :func:`shapely.geometry.buffer`.

        Returns
        -------
        Polygon | MultiPolygon | EmptyRegion
            The extended region.
        """
        return Region2D.from_shapely(self.shapely_object.buffer(distance, **kwargs))


class Region3D(Region):
    """
    Abstract Region class to define the interface for 3-dimensional Region classes.
    """

    @property
    def dimension(self) -> int:
        return 3

    @abstractmethod
    def as_artist(
        self, origin: npt.ArrayLike = (0, 0, 0), **kwargs: Any
    ) -> mpl_patches.Patch:
        """
        Matplotlib patch object for this region (e.g. `matplotlib.patches.Ellipse`).

        Parameters
        ----------
        origin
            The (x, y, z) pixel position of the origin of the displayed image.
            Default is (0, 0, 0).
        kwargs
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        matplotlib.patches.Patch
            Matplotlib patch for the specified region.
        """
        pass

    def plot(self, ax: mpl.axes.Axes | None = None, **kwargs: Any) -> mpl.axes.Axes:
        """
        Provide plot of region as :class:`matplotlib.axes.Axes` object.

        Parameters
        ----------
        ax
            The axes on which to show the image
        kwargs
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        artist = self.as_artist(**kwargs)
        ax.add_artist(artist)

        return ax

    def intersection(self, other: Region) -> Region:
        raise NotImplementedError

    def symmetric_difference(self, other: Region) -> Region:
        raise NotImplementedError

    def union(self, other: Region) -> Region:
        raise NotImplementedError


class RegionND(Region):
    """
    Abstract Region class to define the interface for n-dimensional Region
    classes.
    """

    def intersection(self, other: Region) -> Region:
        raise NotImplementedError

    def symmetric_difference(self, other: Region) -> Region:
        raise NotImplementedError

    def union(self, other: Region) -> Region:
        raise NotImplementedError


class EmptyRegion(Region):
    """
    Region class to define an empty region that has no dimension.
    """

    def __init__(self) -> None:
        self.shapely_object = shPolygon()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def dimension(self) -> None:
        return None

    @property
    def points(self) -> npt.NDArray[np.float_]:
        return np.array([], dtype=np.float_)

    @property
    def centroid(self) -> None:
        return None

    @property
    def max_distance(self) -> int:
        return 0

    @property
    def region_measure(self) -> int:
        return 0

    @property
    def subregion_measure(self) -> int:
        return 0

    @property
    def bounds(self) -> None:
        return None

    @property
    def extent(self) -> None:
        return None

    @property
    def bounding_box(self) -> EmptyRegion:
        return EmptyRegion()

    def intersection(self, other: Region) -> EmptyRegion:
        return EmptyRegion()

    def symmetric_difference(self, other: Region) -> Region:
        return other

    def union(self, other: Region) -> Region:
        return other

    def contains(self, points: npt.ArrayLike) -> npt.NDArray[np.int_]:
        return np.array([], dtype=np.int_)

    def as_artist(self, **kwargs: Any) -> None:
        raise NotImplementedError("EmptyRegion cannot return an artist.")

    def buffer(self, distance: float, **kwargs: Any) -> Region:
        raise NotImplementedError("EmptyRegion cannot be extended.")

    @classmethod
    def from_shapely(cls, shapely_object: shPolygon | shMultiPolygon) -> EmptyRegion:
        if shapely_object.is_empty:
            return cls()
        else:
            raise TypeError("Shapely object must be empty.")


T_Interval = TypeVar("T_Interval", bound="Interval")


class Interval(Region1D):
    """
    Region class to define an interval.

    Parameters
    ----------
    lower_bound
        The lower bound of the interval.
    upper_bound
        The upper bound of the interval.
    """

    def __init__(self, lower_bound: float = 0, upper_bound: float = 1) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._region_specs = (self.lower_bound, self.upper_bound)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.lower_bound}, {self.upper_bound})"

    @classmethod
    def from_intervals(cls: type[T_Interval], intervals: npt.ArrayLike) -> T_Interval:
        """
        Constructor for instantiating Region from list of (min, max) bounds.
        Takes array-like intervals instead of interval to be consistent with
        `Rectangle.from_intervals`.

        Parameters
        ----------
        intervals
            The region bounds for each dimension of shape (2,)

        Returns
        -------
        Interval
        """
        intervals = np.asarray(intervals)
        if np.shape(intervals) != (2,):
            raise TypeError(
                f"Intervals must be of shape (2,) and not {np.shape(intervals)}."
            )
        lower_bound, upper_bound = intervals
        return cls(lower_bound, upper_bound)

    @property
    def lower_bound(self) -> float:
        """
        The lower boundary.

        Returns
        -------
        float
        """
        return self._lower_bound

    @property
    def upper_bound(self) -> float:
        """
        The upper boundary.

        Returns
        -------
        float
        """
        return self._upper_bound

    @property
    def bounds(self) -> npt.NDArray[np.float_]:
        return np.array([self.lower_bound, self.upper_bound])

    @property
    def extent(self) -> npt.NDArray[np.float_]:
        return np.array([abs(self.upper_bound - self.lower_bound)])

    @property
    def intervals(self) -> npt.NDArray[np.float_]:
        """
        Provide bounds in a tuple (min, max) arrangement.

        Returns
        -------
        tuple[tuple[float, float], ...]
            ((min_x, max_x), ...) of shape(dimension, 2).
        """
        return self.bounds

    @property
    def region_specs(self) -> tuple[float, float]:
        """
        Legacy interface to serve legacy RoiRegion.

        Warnings
        --------
        Do not use - will be deprecated.

        Returns
        -------
        tuple[float, float]
        """
        return self._region_specs

    @property
    def points(self) -> npt.NDArray[np.float_]:
        return np.array([self.lower_bound, self.upper_bound])

    @property
    def centroid(self) -> npt.NDArray[np.float_]:
        return np.array([self.lower_bound + (self.upper_bound - self.lower_bound) / 2])

    @property
    def max_distance(self) -> float:
        return self.upper_bound - self.lower_bound

    @property
    def region_measure(self) -> float:
        return self.upper_bound - self.lower_bound

    @property
    def subregion_measure(self) -> int:
        return 0

    @property
    def bounding_box(self) -> Self:
        return self

    def contains(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        points_ = np.asarray(points)
        condition = (points_ >= self.lower_bound) & (points_ < self.upper_bound)
        inside_indices = condition.nonzero()[0]  # points are 1-dimensional
        return inside_indices

    def as_artist(
        self, origin: npt.ArrayLike = (0, 0), **kwargs: Any
    ) -> mpl_patches.Patch:
        raise NotImplementedError

    def buffer(self, distance: float, **kwargs: Any) -> Interval:
        l_bound = self.lower_bound - distance
        u_bound = self.upper_bound + distance
        return Interval(lower_bound=l_bound, upper_bound=u_bound)


T_Rectangle = TypeVar("T_Rectangle", bound="Rectangle")


class Rectangle(Region2D):
    """
    Region class to define a rectangle.

    Parameters
    ----------
    corner : npt.ArrayLike
        A point that defines the lower left corner with shape (2,).
    width : float
        The length of a vector describing the edge in x-direction.
    height : float
        The length of a vector describing the edge in y-direction.
    angle : float
        The angle (in degrees) by which the rectangle is rotated
        counterclockwise around the corner point.
    """

    def __init__(
        self,
        corner: npt.ArrayLike = (0, 0),
        width: float = 1,
        height: float = 1,
        angle: float = 0,
    ) -> None:
        self._corner = np.asarray(corner)
        self._width = width
        self._height = height
        self._angle = angle
        self._region_specs = (corner, width, height, angle)
        self._shapely_object = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({tuple(self.corner)}, {self.width}, {self.height}, {self.angle})"

    def __getattr__(self, attr: str) -> Any:
        """All non-adapted calls are passed to shapely object"""
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self.shapely_object, attr)

    @classmethod
    def from_intervals(
        cls: type[T_Rectangle], intervals: npt.ArrayLike  # noqa: UP006
    ) -> T_Rectangle:
        """
        Constructor for instantiating Region from list of (min, max) bounds.

        Parameters
        ----------
        intervals
            The region bounds for each dimension of shape (2, 2)

        Returns
        -------
        cls
        """
        intervals = np.asarray(intervals)
        if np.shape(intervals) != (2, 2):
            raise TypeError(
                f"Intervals must be of shape (2, 2) and not {np.shape(intervals)}."
            )
        min_x, max_x = intervals[0]
        min_y, max_y = intervals[1]
        corner = (min_x, min_y)
        width = max_x - min_x
        height = max_y - min_y
        angle = 0
        return cls(corner, width, height, angle)

    @property
    def corner(self) -> npt.NDArray[np.float_]:
        """
        A point that defines the lower left corner.

        Returns
        -------
        npt.NDArray[np.float_]
            with shape (2,)
        """
        return self._corner

    @property
    def width(self) -> float:
        """
        The length of a vector describing the edge in x-direction.

        Returns
        -------
        float
        """
        return self._width

    @property
    def height(self) -> float:
        """
        The length of a vector describing the edge in y-direction.

        Returns
        -------
        float
        """
        return self._height

    @property
    def angle(self) -> float:
        """
        The angle (in degrees) by which the rectangle is rotated
        counterclockwise around the corner point.

        Returns
        -------
        float
        """
        return self._angle

    @property
    def intervals(self) -> npt.NDArray[np.float_]:
        """
        Provide bounds in a tuple (min, max) arrangement.

        Returns
        -------
        npt.NDArray[np.float_]
            ((min_x, max_x), ...) of shape(dimension, 2)
        """
        lower_bounds = self.bounds[: self.dimension]
        upper_bounds = self.bounds[self.dimension :]
        return np.array(
            [(lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)]
        )

    @property
    def points(self) -> npt.NDArray[np.float_]:
        rectangle = mpl_patches.Rectangle(
            self.corner,  # type: ignore[arg-type]
            self.width,
            self.height,
            angle=self.angle,
            fill=False,
            edgecolor="b",
            linewidth=1,
        )
        points: npt.NDArray[np.float_] = rectangle.get_verts()  # type: ignore
        return np.array(points[::-1])

    @property
    def region_specs(self):  # type: ignore
        """
        Legacy interface to serve legacy RoiRegion.

        Warnings
        --------
        Do not use - will be deprecated.

        Returns
        -------
        dict
        """
        return self._region_specs

    @property
    def shapely_object(self) -> shPolygon:
        if self._shapely_object is None:
            self._shapely_object = shPolygon(self.points[:-1])
        return self._shapely_object

    @property
    def centroid(self) -> npt.NDArray[np.float_]:
        return np.array(list(self.shapely_object.centroid.coords)[0])

    @property
    def max_distance(self) -> float:
        return_value: float = np.sqrt(self.width**2 + self.height**2)
        return return_value

    @property
    def region_measure(self) -> float:
        return self.width * self.height

    @property
    def subregion_measure(self) -> float:
        return 2 * self.width + 2 * self.height

    def contains(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        points = np.asarray(points)
        if points.size == 0:
            return np.array([], dtype=np.int_)
        polygon_path = mpl_path.Path(self.points, closed=True)
        mask = polygon_path.contains_points(points)
        inside_indices = np.nonzero(mask)[0]
        return inside_indices

    def as_artist(
        self, origin: npt.ArrayLike = (0, 0), **kwargs: Any
    ) -> mpl_patches.Patch:
        origin = np.asarray(origin)
        xy = self.corner[0] - origin[0], self.corner[1] - origin[1]
        return mpl_patches.Rectangle(
            xy=xy, width=self.width, height=self.height, angle=self.angle, **kwargs
        )


class Ellipse(Region2D):
    """
    Region class to define an ellipse.

    Parameters
    ----------
    center : npt.ArrayLike
        A point that defines the center of the ellipse with shape (2,).
    width : float
        The length of a vector describing the principal axis in x-direction
        (before rotation).
    height : float
        The length of a vector describing the principal axis in y-direction
        (before rotation).
    angle : float
        The angle (in degrees) by which the ellipse is rotated
        counterclockwise around the center point.
    """

    def __init__(
        self,
        center: npt.ArrayLike = (0, 0),
        width: float = 1,
        height: float = 1,
        angle: float = 0,
    ) -> None:
        self._center = np.asarray(center)
        self._width = width
        self._height = height
        self._angle = angle
        self._region_specs = (center, width, height, angle)
        self._shapely_object = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({tuple(self.center)}, {self.width}, {self.height}, {self.angle})"

    def __getattr__(self, attr: str) -> Any:
        """All non-adapted calls are passed to shapely object"""
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self.shapely_object, attr)

    @property
    def center(self) -> npt.NDArray[np.float_]:
        """
        A point that defines the center of the ellipse.

        Returns
        -------
        npt.NDArray[np.float_]
            with shape (2,)
        """
        return self._center

    @property
    def width(self) -> float:
        """
        The length of a vector describing the principal axis in x-direction
        (before rotation).

        Returns
        -------
        float
        """
        return self._width

    @property
    def height(self) -> float:
        """
        The length of a vector describing the principal axis in y-direction
        (before rotation).

        Returns
        -------
        float
        """
        return self._height

    @property
    def angle(self) -> float:
        """
        The angle (in degrees) by which the ellipse is rotated
        counterclockwise around the center point.

        Returns
        -------
        float
        """
        return self._angle

    @property
    def points(self) -> npt.NDArray[np.float_]:
        return np.array(self.shapely_object.exterior.coords)[::-1]

    @property
    def region_specs(self):  # type: ignore
        """
        Legacy interface to serve legacy RoiRegion.

        Warnings
        --------
        Do not use - will be deprecated.

        Returns
        -------
        dict
        """
        return self._region_specs

    @property
    def shapely_object(self) -> shPolygon:
        if self._shapely_object is None:
            circle = shPoint((0, 0)).buffer(1)
            ellipse = scale(circle, self.width / 2, self.height / 2)
            rotated_ellipse = rotate(ellipse, self.angle)
            self._shapely_object = translate(rotated_ellipse, *self.center)
        return self._shapely_object

    @property
    def centroid(self) -> npt.NDArray[np.float_]:
        return self.center

    @property
    def max_distance(self) -> float:
        return_value: float = np.max([self.width, self.height])
        return return_value

    @property
    def region_measure(self) -> float:
        return np.pi * self.width / 2 * self.height / 2

    @property
    def subregion_measure(self) -> float:
        # using Ramanujan approximation
        a, b = self.width / 2, self.height / 2
        t = ((a - b) / (a + b)) ** 2
        circumference: float = np.pi * (a + b) * (1 + 3 * t / (10 + np.sqrt(4 - 3 * t)))
        return circumference

    def contains(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        points_ = np.asarray(points)
        if points_.size == 0:
            return np.array([])

        cos_angle = np.cos(np.radians(-self.angle))
        sin_angle = np.sin(np.radians(-self.angle))

        xc = points_[:, 0] - self.center[0]
        yc = points_[:, 1] - self.center[1]

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle

        rad_cc = (xct**2 / (self.width / 2.0) ** 2) + (
            yct**2 / (self.height / 2.0) ** 2
        )

        inside_indices = np.nonzero(rad_cc < 1.0)[0]
        return inside_indices

    def as_artist(
        self, origin: npt.ArrayLike = (0, 0), **kwargs: Any
    ) -> mpl_patches.Patch:
        origin = np.asarray(origin)
        xy = self.center[0] - origin[0], self.center[1] - origin[1]
        return mpl_patches.Ellipse(
            xy=xy, width=self.width, height=self.height, angle=self.angle, **kwargs
        )


class Polygon(Region2D):
    """
    Region class to define a polygon.

    The polygon is constructed from a list of points that can be
    closed (i.e. the first and last point are identical) or
    not (in this case the list of points will be closed implicitly).

    Parameters
    ----------
    points  : npt.ArrayLike
        Points with shape (n_points, 2)
        that define the exterior boundary of a polygon.
    holes  : list[npt.ArrayLike] | None
        Points with shape (n_holes, n_points, 2)
        that define holes within the polygon.
    """

    def __init__(
        self,
        points: npt.ArrayLike = ((0, 0), (0, 1), (1, 1), (1, 0)),
        holes: list[npt.ArrayLike] | None = None,
    ) -> None:
        points = np.asarray(points)
        if np.all(points[0] == points[-1]):
            self._points = np.array(points)
        else:
            self._points = np.append(np.array(points), [points[0]], axis=0)
        if holes is None:
            self._holes = None
        else:
            self._holes = [np.array(hole) for hole in holes]
        self._region_specs = None
        self._shapely_object = None

    def __repr__(self) -> str:
        if self.holes is None:
            return f"{self.__class__.__name__}({self.points.tolist()})"
        else:
            return f"{self.__class__.__name__}({self.points.tolist()}, {[hole.tolist() for hole in self.holes]})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(<self.points>, <self.holes>)"

    def __getattr__(self, attr: str) -> Any:
        """All non-adapted calls are passed to shapely object"""
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self.shapely_object, attr)

    @classmethod
    def from_shapely(cls, polygon: shPolygon) -> Polygon | EmptyRegion:
        if polygon.is_empty:
            return EmptyRegion()
        else:
            points = np.array(polygon.exterior.coords).tolist()
            holes = [
                np.array(interiors.coords).tolist() for interiors in polygon.interiors
            ]
            return cls(points, holes)

    @property
    def points(self) -> npt.NDArray[np.float_]:
        """
        Exterior polygon points.

        Returns
        -------
        npt.NDArray[np.float_]
            of shape(n_points, dimension)
        """
        return self._points

    @property
    def holes(self) -> list[npt.NDArray[np.float_]] | None:
        """
        Holes where each hole is specified by polygon points.

        Returns
        -------
        list[npt.NDArray[np.float_]] | None
            n_holes of shape(n_points, dimension)
        """
        return self._holes

    @property
    def region_specs(self):  # type: ignore
        """
        Legacy interface to serve legacy RoiRegion.

        Warnings
        --------
        Do not use - will be deprecated.

        Returns
        -------
        dict[str, Any]
        """
        return self._region_specs

    @property
    def shapely_object(self) -> shPolygon:
        if self._shapely_object is None:
            self._shapely_object = shPolygon(self.points, self.holes)
        return self._shapely_object

    @property
    def centroid(self) -> npt.NDArray[np.float_]:
        return np.array(list(self.shapely_object.centroid.coords)[0])

    @property
    def max_distance(self) -> float:
        distances = pdist(self.points[:-1])
        return_value: float = np.nanmax(distances)
        return return_value

    @property
    def region_measure(self) -> float:
        return_value: float = self.shapely_object.area
        return return_value

    @property
    def subregion_measure(self) -> float:
        return_value: float = self.shapely_object.length
        return return_value

    def contains(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        _points = np.asarray(points)
        if _points.size == 0:
            return np.array([], dtype=bool)

        # preselect points inside the polygons bounding box to increase performance.
        preselected_points_indices = self.bounding_box.contains(_points)
        if len(preselected_points_indices) == 0:
            return np.array([], dtype=bool)

        points_ = shMultiPoint(_points[preselected_points_indices])
        prepared_polygon = prep(self.shapely_object)
        mask = list(map(prepared_polygon.contains, points_.geoms))
        inside_indices = np.nonzero(mask)[0]
        if len(inside_indices) == 0:
            return np.array([], dtype=bool)
        else:
            return preselected_points_indices[inside_indices]

    def as_artist(
        self, origin: npt.ArrayLike = (0, 0), **kwargs: Any
    ) -> mpl_patches.Patch:
        # todo implement origin
        return mpl_patches.PathPatch(_polygon_path(self.shapely_object), **kwargs)


class MultiPolygon(Region2D):
    """
    Region class to define a region that represents the union of multiple
    polygons.

    Parameters
    ----------
    polygons  : list[Polygon]
        Polygons that define the individual polygons.
    """

    def __init__(self, polygons: list[Polygon]) -> None:
        self._polygons = polygons
        self._region_specs = None
        self._shapely_object = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.polygons})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(<self.polygons>)"

    def __getattr__(self, attr: str) -> Any:
        """All non-adapted calls are passed to shapely object"""
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self.shapely_object, attr)

    @property
    def points(self) -> list[npt.NDArray[np.float_]]:
        """
        Exterior polygon points.

        Returns
        -------
        list[npt.NDArray[np.float_]]
            n_polygons of shape(n_points, dimension)
        """
        return [pol.points for pol in self.polygons]

    @property
    def holes(self) -> list[list[npt.NDArray[np.float_]] | None]:
        """
        Points defining holes.

        Returns
        -------
        list[list[npt.NDArray[np.float_]] | None]
            list of polygon holes
        """
        return [pol.holes for pol in self.polygons]

    @property
    def polygons(self) -> list[Polygon]:
        """
        All polygons that make up the MultiPolygon

        Returns
        -------
        list[Polygon]
        """
        return self._polygons

    @property
    def region_specs(self):  # type: ignore
        """
        Legacy interface to serve legacy RoiRegion.

        Warnings
        --------
        Do not use - will be deprecated.

        Returns
        -------
        dict
        """
        return self._region_specs

    @property
    def shapely_object(self) -> shMultiPolygon:
        if self._shapely_object is None:
            self._shapely_object = shMultiPolygon(
                [pol.shapely_object for pol in self._polygons]
            )
        return self._shapely_object

    @property
    def centroid(self) -> npt.NDArray[np.float_]:
        return np.array(list(self.shapely_object.centroid.coords)[0])

    @property
    def max_distance(self) -> float:
        distances = pdist(np.array([point for pts in self.points for point in pts]))
        return_value: float = np.nanmax(distances)
        return return_value

    @property
    def region_measure(self) -> float:
        return_value: float = self.shapely_object.area
        return return_value

    @property
    def subregion_measure(self) -> float:
        return_value: float = self.shapely_object.length
        return return_value

    def contains(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        if np.asarray(points).size == 0:
            return np.array([], dtype=np.int64)
        points_ = shMultiPoint(points)
        prepared_polygon = prep(self.shapely_object)
        mask = list(map(prepared_polygon.contains, points_.geoms))
        inside_indices = np.nonzero(mask)[0]
        return inside_indices

    def as_artist(
        self, origin: npt.ArrayLike = (0, 0), **kwargs: Any
    ) -> mpl_patches.Patch:
        # todo fix origin
        polygon = shMultiPolygon([pol.shapely_object for pol in self.polygons])
        return mpl_patches.PathPatch(_polygon_path(polygon), **kwargs)

    @classmethod
    def from_shapely(cls, multipolygon: shMultiPolygon) -> MultiPolygon:
        polygons_ = [Polygon.from_shapely(pol) for pol in multipolygon.geoms]
        polygons__ = [pol for pol in polygons_ if not isinstance(pol, EmptyRegion)]
        return cls(polygons__)


T_AxisOrientedCuboid = TypeVar("T_AxisOrientedCuboid", bound="AxisOrientedCuboid")


class AxisOrientedCuboid(Region3D):
    """
    Region class to define an axis-oriented cuboid.

    This is a 3-dimensional convex region with rectangular faces
    and edges that are parallel to coordinate axes.
    Extension in x-, y-, z-coordinates correspond to length, width, height.

    Parameters
    ----------
    corner : npt.ArrayLike
        A point that defines the lower left corner with shape (3,)
    length : float
        The length of a vector describing the edge in x-direction.
    width : float
        The length of a vector describing the edge in y-direction.
    height : float
        The length of a vector describing the edge in z-direction.
    """

    def __init__(
        self,
        corner: npt.ArrayLike = (0, 0, 0),
        length: float = 1,
        width: float = 1,
        height: float = 1,
    ) -> None:
        self._corner = np.asarray(corner)
        self._length = length
        self._width = width
        self._height = height

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({tuple(self.corner)}, "
            f"{self.length}, {self.width}, {self.height})"
        )

    @classmethod
    def from_intervals(
        cls: type[T_AxisOrientedCuboid], intervals: npt.ArrayLike  # noqa: UP006
    ) -> T_AxisOrientedCuboid:
        """
        Constructor for instantiating Region from list of (min, max) bounds.

        Parameters
        ----------
        intervals : npt.ArrayLike
            The region bounds for each dimension of shape (3, 2)

        Returns
        -------
        AxisOrientedCuboid
        """
        intervals = np.asarray(intervals)
        if np.shape(intervals) != (3, 2):
            raise TypeError(
                f"Intervals must be of shape (3, 2) and not {np.shape(intervals)}."
            )
        intervals = np.asarray(intervals)
        min_x, max_x = intervals[0]
        min_y, max_y = intervals[1]
        min_z, max_z = intervals[2]
        corner = (min_x, min_y, min_z)
        length = max_x - min_x
        width = max_y - min_y
        height = max_z - min_z
        return cls(corner, length, width, height)

    @property
    def corner(self) -> npt.NDArray[np.float_]:
        """
        A point that defines the lower left corner.

        Returns
        -------
        npt.NDArray[np.float_]
            with shape (2,)
        """
        return self._corner

    @property
    def length(self) -> float:
        """
        The length of a vector describing the edge in x-direction.

        Returns
        -------
        float
        """
        return self._length

    @property
    def width(self) -> float:
        """
        The length of a vector describing the edge in y-direction.

        Returns
        -------
        float
        """
        return self._width

    @property
    def height(self) -> float:
        """
        The length of a vector describing the edge in z-direction.

        Returns
        -------
        float
        """
        return self._height

    @property
    def points(self) -> npt.NDArray[np.float_]:
        return np.array(list(it.product(*self.intervals)))

    @property
    def centroid(self) -> npt.NDArray[np.float_]:
        return np.array(
            [
                cor + dist / 2
                for cor, dist in zip(
                    self.corner, (self.length, self.width, self.height)
                )
            ]
        )

    @property
    def bounds(self) -> npt.NDArray[np.float_]:
        min_x, min_y, min_z = self.corner
        max_x, max_y, max_z = [  # noqa: UP027
            cor + dist
            for cor, dist in zip(self.corner, (self.length, self.width, self.height))
        ]
        return np.array([min_x, min_y, min_z, max_x, max_y, max_z])

    @property
    def intervals(self) -> npt.NDArray[np.float_]:
        """
        Provide bounds in a tuple (min, max) arrangement.

        Returns
        -------
        tuple[tuple[float, float], ...]
            ((min_x, max_x), ...) of shape(dimension, 2)
        """
        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds
        return np.array([(min_x, max_x), (min_y, max_y), (min_z, max_z)])

    @property
    def extent(self) -> npt.NDArray[np.float_]:
        return np.array([abs(self.length), abs(self.width), abs(self.height)])

    @property
    def bounding_box(self) -> Self:
        return self

    @property
    def max_distance(self) -> float:
        return_value: float = np.sqrt(
            self.length**2 + self.width**2 + self.height**2
        )
        return return_value

    @property
    def region_measure(self) -> float:
        return_value: float = self.length * self.width * self.height
        return return_value

    @property
    def subregion_measure(self) -> float:
        return_value: float = 2 * (
            self.length * self.width
            + self.height * self.width
            + self.height * self.length
        )
        return return_value

    def contains(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        points = np.asarray(points)
        if points.size == 0:
            return np.array([])
        condition_0 = [
            (points[:, i] >= bound) for i, bound in enumerate(self.bounds[:3])
        ]
        condition_1 = [
            (points[:, i] < bound) for i, bound in enumerate(self.bounds[3:])
        ]
        condition = np.all(condition_0, axis=0) & np.all(condition_1, axis=0)
        inside_indices = np.nonzero(condition)[0]
        return inside_indices

    def as_artist(
        self, origin: npt.ArrayLike = (0, 0), **kwargs: Any
    ) -> mpl_patches.Patch:
        raise NotImplementedError

    def buffer(self, distance: float, **kwargs: Any) -> AxisOrientedCuboid:
        mins = self.bounds[: self.dimension]
        maxs = self.bounds[self.dimension :]
        new_mins = [value - distance for value in mins]
        new_maxs = [value + distance for value in maxs]
        return AxisOrientedCuboid(
            new_mins, *(max_ - min_ for max_, min_ in zip(new_maxs, new_mins))
        )


# todo: complete implementation
class Cuboid(Region3D):
    """
    Region class to define a cuboid.

    This is a 3-dimensional convex region with rectangular faces.
    Extension in x-, y-, z-coordinates correspond to length, width, height.
    Corresponding Euler angles are defined by alpha, beta, gamma.

    Parameters
    ----------
    corner : npt.ArrayLike
        A point that defines the lower left corner with shape (2,).
    length : float
        The length of a vector describing the edge in x-direction.
    width : float
        The length of a vector describing the edge in y-direction.
    alpha : float
        The first Euler angle (in degrees) by which the cuboid is rotated.
    beta : float
        The second Euler angle (in degrees) by which the cuboid is rotated.
    gamma : float
        The third Euler angle (in degrees) by which the cuboid is rotated.
    """

    def __init__(
        self,
        corner: npt.ArrayLike = (0, 0, 0),
        length: float = 1,
        width: float = 1,
        height: float = 1,
        alpha: float = 0,
        beta: float = 0,
        gamma: float = 0,
    ) -> None:
        self._corner = np.asarray(corner)
        self._length = length
        self._width = width
        self._height = height
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._region_specs = (corner, length, width, height, alpha, beta, gamma)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({tuple(self.corner)}, "
            f"{self.length}, {self.width}, {self.height}, "
            f"{self.alpha}, {self.beta}, {self.gamma})"
        )

    @property
    def corner(self) -> npt.NDArray[np.float_]:
        """
        A point that defines the lower left corner.

        Returns
        -------
        npt.NDArray[np.float_]
            with shape (2,)
        """
        return self._corner

    @property
    def length(self) -> float:
        """
        The length of a vector describing the edge in x-direction.

        Returns
        -------
        float
        """
        return self._length

    @property
    def width(self) -> float:
        """
        The length of a vector describing the edge in y-direction.

        Returns
        -------
        float
        """
        return self._width

    @property
    def height(self) -> float:
        """
        The length of a vector describing the edge in z-direction.

        Returns
        -------
        float
        """
        return self._height

    @property
    def alpha(self) -> float:
        """
        The first Euler angle (in degrees) by which the cuboid is rotated.

        Returns
        -------
        float
        """
        return self._alpha

    @property
    def beta(self) -> float:
        """
        The sescond Euler angle (in degrees) by which the cuboid is rotated.

        Returns
        -------
        float
        """
        return self._beta

    @property
    def gamma(self) -> float:
        """
        The third Euler angle (in degrees) by which the cuboid is rotated.

        Returns
        -------
        float
        """
        return self._gamma

    @property
    def points(self) -> npt.NDArray[np.float_]:
        raise NotImplementedError

    @property
    def bounds(self) -> npt.NDArray[np.float_]:
        raise NotImplementedError

    @property
    def extent(self) -> npt.NDArray[np.float_]:
        raise NotImplementedError

    @property
    def centroid(self) -> npt.NDArray[np.float_]:
        raise NotImplementedError

    @property
    def max_distance(self) -> float:
        return_value: float = np.sqrt(
            self.length**2 + self.width**2 + self.height**2
        )
        return return_value

    @property
    def region_measure(self) -> float:
        return_value: float = self.length * self.width * self.height
        return return_value

    @property
    def subregion_measure(self) -> float:
        return_value: float = 2 * (
            self.length * self.width
            + self.height * self.width
            + self.height * self.length
        )
        return return_value

    def contains(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        raise NotImplementedError

    def as_artist(
        self, origin: npt.ArrayLike = (0, 0), **kwargs: Any
    ) -> mpl_patches.Patch:
        raise NotImplementedError

    def buffer(self, distance: float, **kwargs: Any) -> Region:
        raise NotImplementedError

    @property
    def bounding_box(self) -> Self:
        return self


T_AxisOrientedHypercuboid = TypeVar(
    "T_AxisOrientedHypercuboid", bound="AxisOrientedHypercuboid"
)


class AxisOrientedHypercuboid(RegionND):
    """
    Region class to define an axis-oriented n-dimensional hypercuboid.

    This is a n-dimensional convex region with rectangular faces
    and edges that are parallel to coordinate axes.
    Extension in x-, y-, z-coordinates correspond to length, width, height.

    Parameters
    ----------
    corner : npt.ArrayLike
        A point with shape (dimension,) that defines the lower left corner.
    lengths : npt.ArrayLike
        Array of shape(dimension,) of length values for the 1-dimensional
        edge vectors.
    """

    def __init__(
        self, corner: npt.ArrayLike = (0, 0, 0), lengths: npt.ArrayLike = (1, 1, 1)
    ):
        corner = np.asarray(corner)
        lengths = np.asarray(lengths)
        if not len(corner) == len(lengths):
            raise TypeError("corner and lengths must have the same dimension.")
        self._corner = corner
        self._lengths = lengths

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({tuple(self.corner)}, {tuple(self.lengths)})"

    @classmethod
    def from_intervals(
        cls: type[T_AxisOrientedHypercuboid], intervals: npt.ArrayLike  # noqa: UP006
    ) -> T_AxisOrientedHypercuboid:
        """
        Constructor for instantiating Region from list of (min, max) bounds.

        Parameters
        ----------
        intervals : npt.ArrayLike
            The region bounds for each dimension of shape (dimension, 2)

        Returns
        -------
        AxisOrientedHypercuboid
        """
        intervals = np.asarray(intervals)
        if np.shape(intervals)[1] != 2:
            raise TypeError(
                f"Intervals must be of shape (dimension, 2) and not {np.shape(intervals)}."
            )
        corner = intervals[:, 0]
        lengths = np.diff(intervals)[:, 0]
        return cls(corner, lengths)

    @property
    def corner(self) -> npt.NDArray[np.float_]:
        """
        A point that defines the lower left corner.

        Returns
        -------
        npt.NDArray[np.float_]
            with shape (dimension,)
        """
        return self._corner

    @property
    def lengths(self) -> npt.NDArray[np.float_]:
        """
        Array of length values for the 1-dimensional edge vectors.

        Returns
        -------
        npt.NDArray[np.float_]
            of shape(dimension,)
        """
        return self._lengths

    @property
    def dimension(self) -> int:
        return len(self.lengths)

    @property
    def intervals(self) -> npt.NDArray[np.float_]:
        """
        Provide bounds in a tuple (min, max) arrangement.

        Returns
        -------
        tuple[tuple[float, float], ...]
            ((min_x, max_x), ...) of shape(dimension, 2).
        """
        return np.array(
            [
                (lower, upper)
                for lower, upper in zip(
                    self.bounds[: self.dimension], self.bounds[self.dimension :]
                )
            ]
        )

    @property
    def points(self) -> npt.NDArray[np.float_]:
        return np.array(list(it.product(*self.intervals)))

    @property
    def bounds(self) -> npt.NDArray[np.float_]:
        return np.concatenate([self.corner, self.corner + self.lengths], axis=0)

    @property
    def extent(self) -> npt.NDArray[np.float_]:
        return np.abs(self.lengths)

    @property
    def centroid(self) -> npt.NDArray[np.float_]:
        return self.corner + self.lengths / 2

    @property
    def max_distance(self) -> float:
        return_value: float = np.sqrt(np.sum(self.lengths**2))
        return return_value

    @property
    def region_measure(self) -> float:
        return_value: float = float(np.prod(self.lengths, dtype=float))
        return return_value

    @property
    def subregion_measure(self) -> float:
        raise NotImplementedError

    @property
    def bounding_box(self) -> Self:
        return self

    def contains(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        points = np.asarray(points)
        if points.size == 0:
            return np.array([])
        condition_0 = [
            (points[:, i] >= bound)
            for i, bound in enumerate(self.bounds[: self.dimension])
        ]
        condition_1 = [
            (points[:, i] < bound)
            for i, bound in enumerate(self.bounds[self.dimension :])
        ]
        condition = np.all(condition_0, axis=0) & np.all(condition_1, axis=0)
        inside_indices = np.nonzero(condition)[0]
        return inside_indices

    def buffer(self, distance: float, **kwargs: Any) -> AxisOrientedHypercuboid:
        # todo: fix round corners or raise warning
        mins = self.bounds[: self.dimension] - distance
        maxs = self.bounds[self.dimension :] + distance
        lengths = maxs - mins
        return AxisOrientedHypercuboid(mins, lengths)


def _polygon_path(polygon: Polygon | MultiPolygon) -> mpl_path.Path:
    """
    Constructs a compound matplotlib path from a Shapely geometric object.
    Adapted from https://pypi.org/project/descartes/
    (BSD license, copyright Sean Gillies)
    """

    def coding(ob: Any) -> npt.NDArray[Any]:
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, "coords", None) or ob)
        vals = np.ones(n, dtype=mpl_path.Path.code_type) * mpl_path.Path.LINETO
        vals[0] = mpl_path.Path.MOVETO
        return vals  # type: ignore

    ptype = polygon.geom_type
    if ptype == "Polygon":
        polygons = [polygon]
    elif ptype == "MultiPolygon":
        polygons = [shPolygon(p) for p in polygon.geoms]
    else:
        raise ValueError("A polygon or multi-polygon representation is required")

    vertices = np.concatenate(
        [
            np.concatenate(
                [np.asarray(t.exterior.coords)[:, :2]]
                + [np.asarray(r.coords)[:, :2] for r in t.interiors]
            )
            for t in polygons
        ]
    )
    codes = np.concatenate(
        [
            np.concatenate([coding(t.exterior)] + [coding(r) for r in t.interiors])
            for t in polygons
        ]
    )

    return mpl_path.Path(vertices, codes)
