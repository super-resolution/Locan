"""

Regions as support for localization data.

This module provides classes to define geometric regions for localization data.
All region classes inherit from the abstract base class `Region`.

"""
# todo: fix docstrings
from __future__ import annotations

import itertools as it
from abc import ABC, abstractmethod

import matplotlib.patches as mpl_patches
import matplotlib.path as mpl_path
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt  # noqa: F401
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

    def __init__(self, region_type, region_specs):
        self.region_specs = region_specs
        self.region_type = region_type

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
            self._region = MultiPolygon(region_specs)

        else:
            raise NotImplementedError(
                f"Region_type {region_type} has not been implemented yet."
            )

    def __getattr__(self, attr):
        """All non-adapted calls are passed to the _region object"""
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self._region, attr)

    def __repr__(self):
        """
        Readable, printable and savable representation of RoiRegion.
        """
        return str(dict(region_type=self.region_type, region_specs=self.region_specs))

    @property
    def region(self):
        return self._region

    @classmethod
    def from_shapely(cls, region_type, shapely_obj):
        if region_type == "shapelyPolygon":
            region_specs = Polygon.from_shapely(shapely_obj).region_specs

        elif region_type == "shapelyMultiPolygon":
            region_specs = MultiPolygon.from_shapely(shapely_obj).region_specs

        else:
            raise NotImplementedError(
                f"Region_type {region_type} has not been implemented yet."
            )

        return cls(region_type=region_type, region_specs=region_specs)

    def contains(self, points) -> npt.NDArray[np.int_]:
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
        return self._region.contains(points)

    def as_artist(self, **kwargs) -> mpl_patches:
        """
        Matplotlib patch object for this region
        (e.g. `matplotlib.patches.Ellipse`).

        Parameters
        ----------
        kwargs : dict
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        matplotlib.patches
            Matplotlib patch for the specified region.
        """
        return self._region.as_artist(**kwargs)

    def to_shapely(self):
        """
        Convert region to a polygon and return as shapely object.

        Returns
        -------
        shapely.Polygon
        """
        return self._region.shapely_object


class Region(ABC):
    """
    Abstract Region class to define the interface for Region-derived classes
    that specify geometric objects to represent regions of interest.
    """

    def __repr__(self):
        return f"{self.__class__.__name__}(...)"

    @classmethod
    def from_intervals(cls, intervals):
        """
        Constructor for instantiating Region from list of (min, max) bounds.
        Takes array-like intervals instead of interval to be consistent with
        `Rectangle.from_intervals`.

        Parameters
        ----------
        intervals : npt.ArrayLike
            The region bounds for each dimension of shape (2,).

        Returns
        -------
        Region
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
            raise TypeError("intervals must be of shape (2,) or (n_features, 2).")

    @property
    @abstractmethod
    def dimension(self):
        """
        The region dimension.

        Returns
        -------
        int
        """
        pass

    @property
    @abstractmethod
    def bounds(self):
        """
        Region bounds min_x, min_y, ..., max_x, max_y, ... for each dimension.

        Returns
        -------
        tuple
            of shape (2 * dimension,)
        """
        pass

    @property
    @abstractmethod
    def extent(self):
        """
        The extent (max_x - min_x), (max_y - min_y), ... for each dimension.

        Returns
        -------
        tuple | npt.NDArray
            of shape (dimension,)
        """
        pass

    @property
    @abstractmethod
    def points(self):
        """
        Point coordinates.

        Returns
        -------
        tuple | npt.NDArray
            of shape (n_points, dimension)
        """
        pass

    @property
    @abstractmethod
    def centroid(self):
        """
        Point coordinates for region centroid.

        Returns
        -------
        tuple | npt.NDArray
            of shape (dimension,)
        """
        pass

    @property
    @abstractmethod
    def max_distance(self):
        """
        The maximum distance between any two points within the region.

        Returns
        -------
        float
        """
        pass

    @property
    @abstractmethod
    def region_measure(self):
        """
        Region measure, i.e. area (for 2d) or volume (for 3d).

        Returns
        -------
        float
        """
        pass

    @property
    @abstractmethod
    def subregion_measure(self):
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
    def bounding_box(self):
        """
        A region describing the minimum axis-aligned bounding box that
        encloses the original region.

        Returns
        -------
        Region
        """
        pass

    @abstractmethod
    def intersection(self, other):
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
    def symmetric_difference(self, other):
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
    def union(self, other):
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

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    @abstractmethod
    def contains(self, points) -> npt.NDArray[np.int_]:
        """
        Return list of indices for all points that are inside the region
        of interest.

        Parameters
        ----------
        points : npt.ArrayLike
            Coordinates of points that are tested for being inside the
            specified region.

        Returns
        -------
        npt.NDArray[np.int_]
            Array with indices for all points in original point array that are
            within the region.
        """
        pass

    def __contains__(self, item):
        return True if list(self.contains([item])) == [0] else False

    @abstractmethod
    def buffer(self, distance):
        """
        Extend the region perpendicular by a `distance`.

        Parameters
        ----------
        distance : float
            Distance by which the region is extended.

        Returns
        -------
        Polygon
            The extended region.
        """
        pass


class Region1D(Region):
    """
    Abstract Region class to define the interface for 1-dimensional Region classes.
    """

    @property
    def dimension(self):
        return 1

    @abstractmethod
    def as_artist(self, origin=(0, 0), **kwargs) -> mpl_patches:
        """
        Matplotlib patch object for this region
        (e.g. `matplotlib.patches.Ellipse`).

        Parameters
        ----------
        origin : npt.ArrayLike
            The (x, y) pixel position of the origin of the displayed image.
            Default is (0, 0).
        kwargs : dict
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        matplotlib.patches
            Matplotlib patch for the specified region.
        """
        pass

    def intersection(self, other):
        raise NotImplementedError

    def symmetric_difference(self, other):
        raise NotImplementedError

    def union(self, other):
        raise NotImplementedError


class Region2D(Region):
    """
    Abstract Region class to define the interface for 2-dimensional
    Region classes.
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_shapely_object"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__["_shapely_object"] = None

    @property
    def dimension(self):
        return 2

    @property
    def bounds(self):
        min_x, min_y, max_x, max_y = self.shapely_object.bounds
        return min_x, min_y, max_x, max_y

    @property
    def extent(self):
        min_x, min_y, max_x, max_y = self.bounds
        return abs(max_x - min_x), abs(max_y - min_y)

    @property
    def bounding_box(self):
        min_x, min_y, max_x, max_y = self.bounds
        return Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 0)

    @property
    @abstractmethod
    def shapely_object(self):
        """
        Geometric object as defined in `shapely`.

        Returns
        -------
        Shapely object
        """
        pass

    @staticmethod
    def from_shapely(shapely_object):
        """
        Constructor for instantiating Region from `shapely` object.

        Parameters
        ----------
        shapely_object : Polygon | MultiPolygon
            Geometric object to be converted into Region

        Returns
        -------
        Region
        """
        ptype = shapely_object.geom_type
        if ptype == "Polygon":
            return Polygon.from_shapely(shapely_object)
        elif ptype == "MultiPolygon":
            return MultiPolygon.from_shapely(shapely_object)

    @abstractmethod
    def as_artist(self, origin=(0, 0), **kwargs) -> mpl_patches:
        """
        Matplotlib patch object for this region
        (e.g. `matplotlib.patches.Ellipse`).

        Parameters
        ----------
        origin : npt.ArrayLike
            The (x, y) pixel position of the origin of the displayed image.
            Default is (0, 0).
        kwargs : dict
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        matplotlib.patches
            Matplotlib patch for the specified region.
        """
        pass

    def plot(self, ax=None, **kwargs):
        """
        Provide plot of region as :class:`matplotlib.axes.Axes` object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to show the image
        kwargs : dict
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

    def intersection(self, other: Region2D):
        shapely_obj = self.shapely_object.intersection(other.shapely_object)
        return Region2D.from_shapely(shapely_obj)

    def symmetric_difference(self, other: Region2D):
        shapely_obj = self.shapely_object.symmetric_difference(other.shapely_object)
        return Region2D.from_shapely(shapely_obj)

    def union(self, other: Region2D):
        shapely_obj = self.shapely_object.union(other.shapely_object)
        return Region2D.from_shapely(shapely_obj)

    def buffer(self, distance, **kwargs):
        """
        Extend the region perpendicular by a `distance`.

        Parameters
        ----------
        distance : float
            Distance by which the region is extended.
        kwargs : dict
            Other parameters passed to :func:`shapely.geometry.buffer`.

        Returns
        -------
        Polygon
            The extended region.
        """
        return Region2D.from_shapely(self.shapely_object.buffer(distance, **kwargs))


class Region3D(Region):
    """
    Abstract Region class to define the interface for 3-dimensional Region classes.
    """

    @property
    def dimension(self):
        return 3

    @abstractmethod
    def as_artist(self, origin=(0, 0), **kwargs) -> mpl_patches:
        """
        Matplotlib patch object for this region (e.g. `matplotlib.patches.Ellipse`).

        Parameters
        ----------
        origin : npt.ArrayLike
            The (x, y) pixel position of the origin of the displayed image.
            Default is (0, 0).
        kwargs : dict
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        matplotlib.patches
            Matplotlib patch for the specified region.
        """
        pass

    def plot(self, ax=None, **kwargs):
        """
        Provide plot of region as :class:`matplotlib.axes.Axes` object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to show the image
        kwargs : dict
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

    def intersection(self, other):
        raise NotImplementedError

    def symmetric_difference(self, other):
        raise NotImplementedError

    def union(self, other):
        raise NotImplementedError


class RegionND(Region):
    """
    Abstract Region class to define the interface for n-dimensional Region
    classes.
    """

    def as_artist(self):
        raise NotImplementedError

    def intersection(self, other):
        raise NotImplementedError

    def symmetric_difference(self, other):
        raise NotImplementedError

    def union(self, other):
        raise NotImplementedError


class EmptyRegion(Region):
    """
    Region class to define an empty region that has no dimension.
    """

    def __init__(self):
        self.shapely_object = shPolygon()

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"{self.__class__.__name__}()"

    @property
    def dimension(self):
        return None

    @property
    def points(self):
        return np.array([])

    @property
    def centroid(self):
        return None

    @property
    def max_distance(self):
        return 0

    @property
    def region_measure(self):
        return 0

    @property
    def subregion_measure(self):
        return 0

    @property
    def bounds(self):
        return None

    @property
    def extent(self):
        return None

    @property
    def bounding_box(self):
        return None

    def intersection(self, other):
        return EmptyRegion()

    def symmetric_difference(self, other):
        return other

    def union(self, other):
        return other

    def contains(self, points):
        return np.array([])

    def as_artist(self, **kwargs):
        raise NotImplementedError("EmptyRegion cannot return an artist.")

    def buffer(self, distance, **kwargs):
        raise NotImplementedError("EmptyRegion cannot be extended.")

    @classmethod
    def from_shapely(cls, polygon):
        if polygon.is_empty:
            return cls()
        else:
            raise TypeError("Shapely object must be empty.")


class Interval(Region1D):
    """
    Region class to define an interval.

    Parameters
    ----------
    lower_bound : float
        The lower bound of the interval.
    upper_bound : float
        The upper bound of the interval.
    """

    def __init__(self, lower_bound=0, upper_bound=1):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._region_specs = (self.lower_bound, self.upper_bound)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.lower_bound}, {self.upper_bound})"

    @classmethod
    def from_intervals(cls, intervals):
        """
        Constructor for instantiating Region from list of (min, max) bounds.
        Takes array-like intervals instead of interval to be consistent with
        `Rectangle.from_intervals`.

        Parameters
        ----------
        intervals : npt.ArrayLike
            The region bounds for each dimension of shape (2,)

        Returns
        -------
        Interval
        """
        lower_bound, upper_bound = intervals
        return cls(lower_bound, upper_bound)

    @property
    def lower_bound(self):
        """
        The lower boundary.

        Returns
        -------
        float
        """
        return self._lower_bound

    @property
    def upper_bound(self):
        """
        The upper boundary.

        Returns
        -------
        float
        """
        return self._upper_bound

    @property
    def bounds(self):
        return self.lower_bound, self.upper_bound

    @property
    def extent(self):
        return abs(self.upper_bound - self.lower_bound)

    @property
    def intervals(self):
        """
        Provide bounds in a tuple (min, max) arrangement.

        Returns
        -------
        tuple
            ((min_x, max_x), ...) of shape(dimension, 2).
        """
        return (self.bounds,)

    @property
    def region_specs(self):
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
    def points(self):
        return np.array([self.lower_bound, self.upper_bound])

    @property
    def centroid(self):
        return self.lower_bound + (self.upper_bound - self.lower_bound) / 2

    @property
    def max_distance(self):
        return self.upper_bound - self.lower_bound

    @property
    def region_measure(self):
        return self.upper_bound - self.lower_bound

    @property
    def subregion_measure(self):
        return None

    @property
    def bounding_box(self):
        return self

    def contains(self, points):
        points_ = np.asarray(points)
        condition = (points_ >= self.lower_bound) & (points_ < self.upper_bound)
        inside_indices = condition.nonzero()[0]  # points are 1-dimensional
        return inside_indices

    def as_artist(self, origin=(0, 0), **kwargs):
        raise NotImplementedError

    def buffer(self, distance, **kwargs):
        l_bound = self.lower_bound - distance
        u_bound = self.upper_bound + distance
        return Interval(lower_bound=l_bound, upper_bound=u_bound)


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

    def __init__(self, corner=(0, 0), width=1, height=1, angle=0):
        self._corner = corner
        self._width = width
        self._height = height
        self._angle = angle
        self._region_specs = (corner, width, height, angle)
        self._shapely_object = None

    def __repr__(self):
        return f"{self.__class__.__name__}({tuple(self.corner)}, {self.width}, {self.height}, {self.angle})"

    def __getattr__(self, attr):
        """All non-adapted calls are passed to shapely object"""
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self.shapely_object, attr)

    @classmethod
    def from_intervals(cls, intervals):
        """
        Constructor for instantiating Region from list of (min, max) bounds.

        Parameters
        ----------
        intervals : tuple | list | npt.ArrayLike
            The region bounds for each dimension of shape (2, 2)

        Returns
        -------
        Rectangle
        """
        min_x, max_x = intervals[0]
        min_y, max_y = intervals[1]
        corner = (min_x, min_y)
        width = max_x - min_x
        height = max_y - min_y
        angle = 0
        return cls(corner, width, height, angle)

    @property
    def corner(self) -> npt.NDArray:
        """
        A point that defines the lower left corner.

        Returns
        -------
        npt.NDArray
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
    def intervals(self) -> tuple:
        """
        Provide bounds in a tuple (min, max) arrangement.

        Returns
        -------
        tuple
            ((min_x, max_x), ...) of shape(dimension, 2)
        """
        lower_bounds = self.bounds[: self.dimension]
        upper_bounds = self.bounds[self.dimension :]
        return tuple(
            ((lower, upper) for lower, upper in zip(lower_bounds, upper_bounds))
        )

    @property
    def points(self):
        rectangle = mpl_patches.Rectangle(
            self.corner,
            self.width,
            self.height,
            angle=self.angle,
            fill=False,
            edgecolor="b",
            linewidth=1,
        )
        points = rectangle.get_verts()
        return points[::-1]

    @property
    def region_specs(self):
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
    def shapely_object(self):
        if self._shapely_object is None:
            self._shapely_object = shPolygon(self.points[:-1])
        return self._shapely_object

    @property
    def centroid(self):
        return list(self.shapely_object.centroid.coords)[0]

    @property
    def max_distance(self):
        return np.sqrt(self.width**2 + self.height**2)

    @property
    def region_measure(self):
        return self.width * self.height

    @property
    def subregion_measure(self):
        return 2 * self.width + 2 * self.height

    def contains(self, points):
        if np.asarray(points).size == 0:
            return np.array([])
        polygon_path = mpl_path.Path(self.points, closed=True)
        mask = polygon_path.contains_points(points)
        inside_indices = np.nonzero(mask)[0]
        return inside_indices

    def as_artist(self, origin=(0, 0), **kwargs):
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

    def __init__(self, center=(0, 0), width=1, height=1, angle=0):
        self._center = center
        self._width = width
        self._height = height
        self._angle = angle
        self._region_specs = (center, width, height, angle)
        self._shapely_object = None

    def __repr__(self):
        return f"{self.__class__.__name__}({tuple(self.center)}, {self.width}, {self.height}, {self.angle})"

    def __getattr__(self, attr):
        """All non-adapted calls are passed to shapely object"""
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self.shapely_object, attr)

    @property
    def center(self):
        """
        A point that defines the center of the ellipse.

        Returns
        -------
        npt.NDArray
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
    def points(self):
        return np.array(self.shapely_object.exterior.coords)[::-1]

    @property
    def region_specs(self):
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
    def shapely_object(self):
        if self._shapely_object is None:
            circle = shPoint((0, 0)).buffer(1)
            ellipse = scale(circle, self.width / 2, self.height / 2)
            rotated_ellipse = rotate(ellipse, self.angle)
            self._shapely_object = translate(rotated_ellipse, *self.center)
        return self._shapely_object

    @property
    def centroid(self):
        return self.center

    @property
    def max_distance(self):
        return np.max([self.width, self.height])

    @property
    def region_measure(self):
        return np.pi * self.width / 2 * self.height / 2

    @property
    def subregion_measure(self):
        # using Ramanujan approximation
        a, b = self.width / 2, self.height / 2
        t = ((a - b) / (a + b)) ** 2
        circumference = np.pi * (a + b) * (1 + 3 * t / (10 + np.sqrt(4 - 3 * t)))
        return circumference

    def contains(self, points):
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

    def as_artist(self, origin=(0, 0), **kwargs):
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
    holes  : npt.ArrayLike
        Points with shape (n_holes, n_points, 2)
        that define holes within the polygon.
    """

    def __init__(self, points=((0, 0), (0, 1), (1, 1), (1, 0)), holes=None):
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

    def __repr__(self):
        if self.holes is None:
            return f"{self.__class__.__name__}({self.points.tolist()})"
        else:
            return f"{self.__class__.__name__}({self.points.tolist()}, {[hole.tolist() for hole in self.holes]})"

    def __str__(self):
        return f"{self.__class__.__name__}(<self.points>, <self.holes>)"

    def __getattr__(self, attr):
        """All non-adapted calls are passed to shapely object"""
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self.shapely_object, attr)

    @classmethod
    def from_shapely(cls, polygon):
        if polygon.is_empty:
            return EmptyRegion()
        else:
            points = np.array(polygon.exterior.coords).tolist()
            holes = [
                np.array(interiors.coords).tolist() for interiors in polygon.interiors
            ]
            return cls(points, holes)

    @property
    def points(self) -> npt.NDArray:
        """
        Exterior polygon points.

        Returns
        -------
        npt.NDArray
            of shape(n_points, dimension)
        """
        return self._points

    @property
    def holes(self):
        """
        Holes where each hole is specified by polygon points.

        Returns
        -------
        list[npt.NDArray]
            n_holes of shape(n_points, dimension)
        """
        return self._holes

    @property
    def region_specs(self):
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
    def shapely_object(self):
        if self._shapely_object is None:
            self._shapely_object = shPolygon(self.points, self.holes)
        return self._shapely_object

    @property
    def centroid(self):
        return list(self.shapely_object.centroid.coords)[0]

    @property
    def max_distance(self):
        distances = pdist(self.points[:-1])
        return np.nanmax(distances)

    @property
    def region_measure(self):
        return self.shapely_object.area

    @property
    def subregion_measure(self):
        return self.shapely_object.length

    def contains(self, points):
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

    def as_artist(self, **kwargs):
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

    def __init__(self, polygons):
        self._polygons = polygons
        self._region_specs = None
        self._shapely_object = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.polygons})"

    def __str__(self):
        return f"{self.__class__.__name__}(<self.polygons>)"

    def __getattr__(self, attr):
        """All non-adapted calls are passed to shapely object"""
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self.shapely_object, attr)

    @property
    def points(self):
        """
        Exterior polygon points.

        Returns
        -------
        list[npt.NDArray]
            n_polygons of shape(n_points, dimension)
        """
        return [pol.points for pol in self.polygons]

    @property
    def holes(self):
        """
        Points defining holes.

        Returns
        -------
        list
            list of polygon holes
        """
        return [pol.holes for pol in self.polygons]

    @property
    def polygons(self):
        """
        All polygons that make up the MultiPolygon

        Returns
        -------
        list[Polygon]
        """
        return self._polygons

    @property
    def region_specs(self):
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
    def shapely_object(self):
        if self._shapely_object is None:
            self._shapely_object = shMultiPolygon(
                [pol.shapely_object for pol in self._polygons]
            )
        return self._shapely_object

    @property
    def centroid(self):
        return list(self.shapely_object.centroid.coords)[0]

    @property
    def max_distance(self):
        distances = pdist(np.array([point for pts in self.points for point in pts]))
        return np.nanmax(distances)

    @property
    def region_measure(self):
        return self.shapely_object.area

    @property
    def subregion_measure(self):
        return self.shapely_object.length

    def contains(self, points):
        if np.asarray(points).size == 0:
            return np.array([])
        points_ = shMultiPoint(points)
        prepared_polygon = prep(self.shapely_object)
        mask = list(map(prepared_polygon.contains, points_.geoms))
        inside_indices = np.nonzero(mask)[0]
        return inside_indices

    def as_artist(self, **kwargs):
        polygon = shMultiPolygon([pol.shapely_object for pol in self.polygons])
        return mpl_patches.PathPatch(_polygon_path(polygon), **kwargs)

    @classmethod
    def from_shapely(cls, multipolygon):
        return cls([Polygon.from_shapely(pol) for pol in multipolygon.geoms])


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

    def __init__(self, corner=(0, 0, 0), length=1, width=1, height=1):
        self._corner = corner
        self._length = length
        self._width = width
        self._height = height

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({tuple(self.corner)}, "
            f"{self.length}, {self.width}, {self.height})"
        )

    @classmethod
    def from_intervals(cls, intervals):
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
        min_x, max_x = intervals[0]
        min_y, max_y = intervals[1]
        min_z, max_z = intervals[2]
        corner = (min_x, min_y, min_z)
        length = max_x - min_x
        width = max_y - min_y
        height = max_z - min_z
        return cls(corner, length, width, height)

    @property
    def corner(self) -> npt.NDArray:
        """
        A point that defines the lower left corner.

        Returns
        -------
        npt.NDArray
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
    def points(self):
        return tuple(it.product(*self.intervals))

    @property
    def centroid(self):
        return tuple(
            (
                cor + dist / 2
                for cor, dist in zip(
                    self.corner, (self.length, self.width, self.height)
                )
            )
        )

    @property
    def bounds(self):
        min_x, min_y, min_z = self.corner
        max_x, max_y, max_z = [  # noqa: UP027
            cor + dist
            for cor, dist in zip(self.corner, (self.length, self.width, self.height))
        ]
        return min_x, min_y, min_z, max_x, max_y, max_z

    @property
    def intervals(self) -> tuple:
        """
        Provide bounds in a tuple (min, max) arrangement.

        Returns
        -------
        tuple
            ((min_x, max_x), ...) of shape(dimension, 2)
        """
        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds
        return ((min_x, max_x), (min_y, max_y), (min_z, max_z))

    @property
    def extent(self):
        return abs(self.length), abs(self.width), abs(self.height)

    @property
    def bounding_box(self):
        return self

    @property
    def max_distance(self):
        return np.sqrt(self.length**2 + self.width**2 + self.height**2)

    @property
    def region_measure(self):
        return self.length * self.width * self.height

    @property
    def subregion_measure(self):
        return 2 * (
            self.length * self.width
            + self.height * self.width
            + self.height * self.length
        )

    def contains(self, points):
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

    def as_artist(self, origin=(0, 0), **kwargs):
        raise NotImplementedError

    def buffer(self, distance, **kwargs):
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
        self, corner=(0, 0, 0), length=1, width=1, height=1, alpha=0, beta=0, gamma=0
    ):
        self._corner = corner
        self._length = length
        self._width = width
        self._height = height
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._region_specs = (corner, length, width, height, alpha, beta, gamma)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({tuple(self.corner)}, "
            f"{self.length}, {self.width}, {self.height}, "
            f"{self.alpha}, {self.beta}, {self.gamma})"
        )

    @property
    def corner(self) -> npt.NDArray:
        """
        A point that defines the lower left corner.

        Returns
        -------
        npt.NDArray
            with shape (2,)
        """
        return self._corner

    @property
    def length(self):
        """
        The length of a vector describing the edge in x-direction.

        Returns
        -------
        float
        """
        return self._length

    @property
    def width(self):
        """
        The length of a vector describing the edge in y-direction.

        Returns
        -------
        float
        """
        return self._width

    @property
    def height(self):
        """
        The length of a vector describing the edge in z-direction.

        Returns
        -------
        float
        """
        return self._height

    @property
    def alpha(self):
        """
        The first Euler angle (in degrees) by which the cuboid is rotated.

        Returns
        -------
        float
        """
        return self._alpha

    @property
    def beta(self):
        """
        The sescond Euler angle (in degrees) by which the cuboid is rotated.

        Returns
        -------
        float
        """
        return self._beta

    @property
    def gamma(self):
        """
        The third Euler angle (in degrees) by which the cuboid is rotated.

        Returns
        -------
        float
        """
        return self._gamma

    @property
    def points(self):
        raise NotImplementedError

    @property
    def bounds(self):
        raise NotImplementedError

    @property
    def extent(self):
        raise NotImplementedError

    @property
    def centroid(self):
        raise NotImplementedError

    @property
    def max_distance(self):
        return np.sqrt(self.length**2 + self.width**2 + self.height**2)

    @property
    def region_measure(self):
        return self.length * self.width * self.height

    @property
    def subregion_measure(self):
        return 2 * (
            self.length * self.width
            + self.height * self.width
            + self.height * self.length
        )

    def contains(self, points):
        raise NotImplementedError

    def as_artist(self, origin=(0, 0), **kwargs):
        raise NotImplementedError

    def buffer(self, distance, **kwargs):
        raise NotImplementedError

    @property
    def bounding_box(self):
        return self


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

    def __init__(self, corner=(0, 0, 0), lengths=(1, 1, 1)):
        if not len(corner) == len(lengths):
            raise TypeError("corner and lengths must have the same dimension.")
        self._corner = np.array(corner)
        self._lengths = np.array(lengths)

    def __repr__(self):
        return f"{self.__class__.__name__}({tuple(self.corner)}, {tuple(self.lengths)})"

    @classmethod
    def from_intervals(cls, intervals):
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
        intervals = np.array(intervals)
        corner = intervals[:, 0]
        lengths = np.diff(intervals)[:, 0]
        return cls(corner, lengths)

    @property
    def corner(self) -> npt.NDArray:
        """
        A point that defines the lower left corner.

        Returns
        -------
        npt.NDArray
            with shape (dimension,)
        """
        return self._corner

    @property
    def lengths(self) -> npt.NDArray:
        """
        Array of length values for the 1-dimensional edge vectors.

        Returns
        -------
        npt.NDArray
            of shape(dimension,)
        """
        return self._lengths

    @property
    def dimension(self):
        return len(self.lengths)

    @property
    def intervals(self):
        """
        Provide bounds in a tuple (min, max) arrangement.

        Returns
        -------
        tuple
            ((min_x, max_x), ...) of shape(dimension, 2).
        """
        return tuple(
            (lower, upper)
            for lower, upper in zip(
                self.bounds[: self.dimension], self.bounds[self.dimension :]
            )
        )

    @property
    def points(self):
        return tuple(it.product(*self.intervals))

    @property
    def bounds(self):
        return np.concatenate([self.corner, self.corner + self.lengths], axis=0)

    @property
    def extent(self):
        return np.abs(self.lengths)

    @property
    def centroid(self):
        return self.corner + self.lengths / 2

    @property
    def max_distance(self):
        return np.sqrt(np.sum(self.lengths**2))

    @property
    def region_measure(self):
        return np.prod(self.lengths)

    @property
    def subregion_measure(self):
        raise NotImplementedError

    @property
    def bounding_box(self):
        return self

    def contains(self, points):
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

    def buffer(self, distance):
        mins = self.bounds[: self.dimension] - distance
        maxs = self.bounds[self.dimension :] + distance
        lengths = maxs - mins
        return AxisOrientedHypercuboid(mins, lengths)


def _polygon_path(polygon):
    """
    Constructs a compound matplotlib path from a Shapely geometric object.
    Adapted from https://pypi.org/project/descartes/
    (BSD license, copyright Sean Gillies)
    """

    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, "coords", None) or ob)
        vals = np.ones(n, dtype=mpl_path.Path.code_type) * mpl_path.Path.LINETO
        vals[0] = mpl_path.Path.MOVETO
        return vals

    ptype = polygon.geom_type
    if ptype == "Polygon":
        polygon = [polygon]
    elif ptype == "MultiPolygon":
        polygon = [shPolygon(p) for p in polygon.geoms]
    else:
        raise ValueError("A polygon or multi-polygon representation is required")

    vertices = np.concatenate(
        [
            np.concatenate(
                [np.asarray(t.exterior.coords)[:, :2]]
                + [np.asarray(r.coords)[:, :2] for r in t.interiors]
            )
            for t in polygon
        ]
    )
    codes = np.concatenate(
        [
            np.concatenate([coding(t.exterior)] + [coding(r) for r in t.interiors])
            for t in polygon
        ]
    )

    return mpl_path.Path(vertices, codes)
