"""

Regions as support for localization data.

This module provides classes to define regions for localization data.
All region classes inherit from the abstract base class `Region`.

"""
# todo: fix docstrings

from abc import ABC, abstractmethod
import itertools as it

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mPath
import matplotlib.patches as mPatches
from scipy.spatial.distance import pdist
from shapely.geometry import asPoint, asMultiPoint
from shapely.geometry import Polygon as shPolygon
from shapely.geometry import MultiPolygon as shMultiPolygon
from shapely.prepared import prep
from shapely.affinity import scale, rotate, translate


__all__ = ['Region', 'Region1D', 'Region2D', 'Region3D', 'RegionND', 'EmptyRegion',
           'Interval',
           'Rectangle', 'Ellipse', 'Polygon', 'MultiPolygon',
           'AxisOrientedCuboid', 'Cuboid',
           'AxisOrientedHypercuboid'
           ]

# __all__ += ['Ellipsoid' 'Polyhedron']
# __all__ += ['Polytope']

__all__ += ['RoiRegion']


class RoiRegion:
    """
    Region object to specify regions of interest.

    A region that defines a region of interest with methods for getting a printable representation (that can also be
    saved in a yaml file), for returning a matplotlib patch that can be shown in a graph, for finding points within
    the region.

    Parameters
    ----------
    region_type : str
        A string indicating the roi shape.
        In 1D it can be `interval`.
        In 2D it can be either `rectangle`, `ellipse`, or closed `polygon`.
        In 2D it can also be `shapelyPolygon` or `shapelyMultiPolygon`.
        In 3D it can be either `cuboid` or `ellipsoid` or `polyhedron` (not implemented yet).
    region_specs : tuple
        1D rois are defined by the following tuple:
        * interval: (start, stop)
        2D rois are defined by the following tuples:
        * rectangle: ((corner_x, corner_y), width, height, angle) with angle in degree
        * ellipse: ((center_x, center_y), width, height, angle) with angle in degree
        * polygon: ((point1_x, point1_y), (point2_x, point2_y), ..., (point1_x, point1_y))
        * shapelyPolygon: ((point_tuples), ((hole_tuples), ...))
        * shapelyMultiPolygon: (shapelyPolygon_specs_1, shapelyPolygon_specs_2, ...)
        3D rois are defined by the following tuples:
        * cuboid: ((corner_x, corner_y, corner_z), length, width, height, angle_1, angle_2, angle_3)
        * ellipsoid: ((center_x, center_y, center_z), length, width, height, angle_1, angle_2, angle_3)
        * polyhedron: (...)

    Attributes
    ----------
    region_type : str
        Type of region
    region_specs : tuple
        Specifications for region
    _region : RoiRegion
        RoiRegion instance for the specified region type.
    polygon : numpy.ndarray of tuples
        Array of points for a closed polygon approximating the region of interest in clockwise orientation. The first
        and last point must be identical.
    dimension : int
        Spatial dimension of region
    centroid : tuple of float
        Centroid coordinates
    max_distance : array-like of float
        Maximum distance between any two points in the region
    region_measure : float
        Hull measure, i.e. area or volume
    subregion_measure : float
        Measure of the sub-dimensional region, i.e. circumference or surface.
    """

    def __init__(self, region_type, region_specs):
        self.region_specs = region_specs
        self.region_type = region_type

        if region_type == 'interval':
            self._region = Interval(*region_specs)

        elif region_type == 'rectangle':
            self._region = Rectangle(*region_specs)

        elif region_type == 'ellipse':
            self._region = Ellipse(*region_specs)

        elif region_type == 'polygon':
            self._region = Polygon(region_specs)

        elif region_type == 'shapelyPolygon':
            self._region = Polygon(region_specs)

        elif region_type == 'shapelyMultiPolygon':
            self._region = MultiPolygon(region_specs)

        else:
            raise NotImplementedError(f'Region_type {region_type} has not been implemented yet.')

    def __getattr__(self, attr):
        """All non-adapted calls are passed to the _region object"""
        if attr.startswith('__') and attr.endswith('__'):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self._region, attr)

    def __repr__(self):
        """
        Readable, printable and savable representation of RoiRegion.
        """
        return str(dict(region_type=self.region_type, region_specs=self.region_specs))

    @classmethod
    def from_shapely(cls, region_type, shapely_obj):
        if region_type == 'shapelyPolygon':
            region_specs = Polygon.from_shapely(shapely_obj).region_specs

        elif region_type == 'shapelyMultiPolygon':
            region_specs = MultiPolygon.from_shapely(shapely_obj).region_specs

        else:
            raise NotImplementedError(f'Region_type {region_type} has not been implemented yet.')

        return cls(region_type=region_type, region_specs=region_specs)

    def contains(self, points):
        """
        Return list of indices for all points that are inside the region of interest.

        Parameters
        ----------
        points : array
            2D or 3D coordinates of oints that are tested for being inside the specified region.

        Returns
        -------
        numpy.ndarray of ints
            Array with indices for all points in original point array that are within the region.
        """
        return self._region.contains(points)

    def as_artist(self, **kwargs):
        """
        Matplotlib patch object for this region (e.g. `matplotlib.patches.Ellipse`).

        Parameters
        ----------
        kwargs : dict
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        patch : matplotlib.patches
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
       Region object to specify geometric objects that represent regions of interest.

       Attributes
       ----------
       dimension : int
           Spatial dimension of region
       points : numpy.ndarray of tuples
           Array of points for a closed polygon approximating the region of interest in clockwise orientation. The first
           and last point are identical.
       centroid : tuple of float
           Centroid coordinates
       max_distance : array-like of float
           Maximum distance between any two points in the region
       region_measure : float
           Hull measure, i.e. area or volume
       subregion_measure : float
           Measure of the sub-dimensional region, i.e. circumference or surface.
       """
    def __repr__(self):
        return f'{self.__class__.__name__}(...)'

    @classmethod
    def from_intervals(cls, intervals):
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
        """Region dimension"""
        pass

    @property
    @abstractmethod
    def points(self):
        """Tuple of point coordinates."""
        pass

    @property
    @abstractmethod
    def centroid(self):
        """Tuple of coordinates for region centroid."""
        pass

    @property
    @abstractmethod
    def max_distance(self):
        """The maximum distance between any two points within the region."""
        pass

    @property
    @abstractmethod
    def region_measure(self):
        pass

    @property
    @abstractmethod
    def subregion_measure(self):
        pass

    @property
    @abstractmethod
    def bounds(self):
        """Tuple of min_x, min_y, ..., max_x, max_y, ..."""
        pass

    @property
    @abstractmethod
    def extent(self):
        """Tuple of absolute differences for bounds in each dimension."""
        pass

    @property
    @abstractmethod
    def bounding_box(self):
        pass

    @abstractmethod
    def intersection(self, other):
        """
        Returns a region representing the intersection of this region with
        ``other``.
        """
        raise NotImplementedError

    @abstractmethod
    def symmetric_difference(self, other):
        """
        Returns the union of the two regions minus any areas contained in the
        intersection of the two regions.
        """
        raise NotImplementedError

    @abstractmethod
    def union(self, other):
        """
        Returns a region representing the union of this region with ``other``.
        """
        raise NotImplementedError

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    @abstractmethod
    def contains(self, points):
        """
        Return list of indices for all points that are inside the region of interest.

        Parameters
        ----------
        points : array-like
            Coordinates of points that are tested for being inside the specified region.

        Returns
        -------
        numpy.ndarray of ints
            Array with indices for all points in original point array that are within the region.
        """
        pass

    @abstractmethod
    def as_artist(self, origin=(0, 0), **kwargs):
        """
        Matplotlib patch object for this region (e.g. `matplotlib.patches.Ellipse`).

        Parameters
        ----------
        origin : array_like
            The (x, y) pixel position of the origin of the displayed image.
            Default is (0, 0).
        kwargs : dict
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        patch : matplotlib.patches
            Matplotlib patch for the specified region.
        """
        pass

    @abstractmethod
    def buffer(self, distance, **kwargs):
        """
        Extend the region perpendicular by a distance.

        Parameters
        ----------
        distance : float
            Distance by which the region is extended.
        kwargs : dict
            Other parameters passed to :func:`shapely.geometry.buffer` for :class:`Region2D`.

        Returns
        -------
        Polygon
            The extended region.
        """
        pass


class Region1D(Region):

    def intersection(self, other):
        """
        Returns a region representing the intersection of this region with
        ``other``.
        """
        raise NotImplementedError

    def symmetric_difference(self, other):
        """
        Returns the union of the two regions minus any areas contained in the
        intersection of the two regions.
        """
        raise NotImplementedError

    def union(self, other):
        """
        Returns a region representing the union of this region with ``other``.
        """
        raise NotImplementedError


class Region2D(Region):

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_shapely_object']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__['_shapely_object'] = None

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
        pass

    @staticmethod
    def from_shapely(shapely_object):
        ptype = shapely_object.geom_type
        if ptype == 'Polygon':
            return Polygon.from_shapely(shapely_object)
        elif ptype == 'MultiPolygon':
            return MultiPolygon.from_shapely(shapely_object)

    def intersection(self, other):
        """
        Returns a region representing the intersection of this region with
        ``other``.
        """
        shapely_obj = self.shapely_object.intersection(other.shapely_object)
        return Region2D.from_shapely(shapely_obj)

    def symmetric_difference(self, other):
        """
        Returns the union of the two regions minus any areas contained in the
        intersection of the two regions.
        """
        shapely_obj = self.shapely_object.symmetric_difference(other.shapely_object)
        return Region2D.from_shapely(shapely_obj)

    def union(self, other):
        """
        Returns a region representing the union of this region with ``other``.
        """
        shapely_obj = self.shapely_object.union(other.shapely_object)
        return Region2D.from_shapely(shapely_obj)

    def buffer(self, distance, **kwargs):
        """
        Extend region using the shapely buffer method.

        Parameters
        ----------
        distance
        kwargs

        Returns
        -------

        """
        return Region2D.from_shapely(self.shapely_object.buffer(distance, **kwargs))

    def plot(self, ax=None, **kwargs):
        """
        Provide plot of region as :class:`matplotlib.axes.Axes` object.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which to show the image
        kwargs : dict
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        artist = self.as_artist(**kwargs)
        ax.add_artist(artist)

        return ax


class Region3D(Region):

    def intersection(self, other):
        """
        Returns a region representing the intersection of this region with
        ``other``.
        """
        raise NotImplementedError

    def symmetric_difference(self, other):
        """
        Returns the union of the two regions minus any areas contained in the
        intersection of the two regions.
        """
        raise NotImplementedError

    def union(self, other):
        """
        Returns a region representing the union of this region with ``other``.
        """
        raise NotImplementedError


class RegionND(Region):
    pass


class EmptyRegion(Region):

    def __init__(self):
        self.shapely_object = shPolygon()

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'{self.__class__.__name__}()'

    def __getattr__(self, attr):
        """All non-adapted calls are passed to the object"""
        if attr.startswith('__') and attr.endswith('__'):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self.shapely_object, attr)

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
        """
        Returns a region representing the intersection of this region with
        ``other``.
        """
        return EmptyRegion()

    def symmetric_difference(self, other):
        """
        Returns the union of the two regions minus any areas contained in the
        intersection of the two regions.
        """
        return other

    def union(self, other):
        """
        Returns a region representing the union of this region with ``other``.
        """
        return other

    def contains(self, points):
        return np.array([])

    def as_artist(self, **kwargs):
        raise NotImplementedError("EmptyRegion cannot return an artist.")

    def buffer(self, distance, **kwargs):
        raise NotImplementedError("An empty region cannot be extended.")

    @classmethod
    def from_shapely(cls, polygon):
        if polygon.is_empty:
            return cls()
        else:
            raise TypeError("polygon must be empty.")


class Interval(Region1D):

    def __init__(self, lower_bound=0, upper_bound=1):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._region_specs = (self.lower_bound, self.upper_bound)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.lower_bound}, {self.upper_bound})'

    @classmethod
    def from_intervals(cls, intervals):
        # using intervals instead of interval to be consistent with Rectangle.from_intervals
        lower_bound, upper_bound = intervals
        return cls(lower_bound, upper_bound)

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def region_specs(self):
        return self._region_specs

    @property
    def dimension(self):
        return 1

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
    def bounds(self):
        return self.lower_bound, self.upper_bound

    @property
    def intervals(self):
        return (self.bounds,)

    @property
    def extent(self):
        return abs(self.upper_bound - self.lower_bound)

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

    def __init__(self, corner=(0, 0), width=1, height=1, angle=0):
        """ rectangle: ((corner_x, corner_y), width, height, angle) with angle in degree"""
        self._corner = corner
        self._width = width
        self._height = height
        self._angle = angle
        self._region_specs = (corner, width, height, angle)
        self._shapely_object = None

    def __repr__(self):
        return f'{self.__class__.__name__}({tuple(self.corner)}, {self.width}, {self.height}, {self.angle})'

    @classmethod
    def from_intervals(cls, intervals):
        min_x, max_x = intervals[0]
        min_y, max_y = intervals[1]
        corner = (min_x, min_y)
        width = max_x - min_x
        height = max_y - min_y
        angle = 0
        return cls(corner, width, height, angle)

    @property
    def corner(self):
        return self._corner

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def angle(self):
        return self._angle

    @property
    def intervals(self):
        """Tuple of ((min_x, max_x), (min_y, max_y))."""
        lower_bounds = self.bounds[:self.dimension]
        upper_bounds = self.bounds[self.dimension:]
        return tuple(((lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)))

    @property
    def region_specs(self):
        return self._region_specs

    @property
    def shapely_object(self):
        if self._shapely_object is None:
            self._shapely_object = shPolygon(self.points[:-1])
        return self._shapely_object

    @property
    def dimension(self):
        return 2

    @property
    def points(self):
        rectangle = mPatches.Rectangle(self.corner, self.width, self.height, angle=self.angle,
                                       fill=False, edgecolor='b', linewidth=1)
        points = rectangle.get_verts()
        return points[::-1]

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
        polygon_path = mPath.Path(self.points, closed=True)
        mask = polygon_path.contains_points(points)
        inside_indices = np.nonzero(mask)[0]
        return inside_indices

    def as_artist(self, origin=(0, 0), **kwargs):
        xy = self.corner[0] - origin[0], self.corner[1] - origin[1]
        return mPatches.Rectangle(xy=xy, width=self.width, height=self.height, angle=self.angle, **kwargs)


class Ellipse(Region2D):

    def __init__(self, center=(0, 0), width=1, height=1, angle=0):
        """ellipse: ((center_x, center_y), width, height, angle) with angle in degree"""
        self._center = center
        self._width = width
        self._height = height
        self._angle = angle
        self._region_specs = (center, width, height, angle)
        self._shapely_object = None

    def __repr__(self):
        return f'{self.__class__.__name__}({tuple(self.center)}, {self.width}, {self.height}, {self.angle})'

    @property
    def center(self):
        return self._center

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def angle(self):
        return self._angle

    @property
    def region_specs(self):
        return self._region_specs

    @property
    def dimension(self):
        return 2

    @property
    def shapely_object(self):
        if self._shapely_object is None:
            circle = asPoint((0, 0)).buffer(1)
            ellipse = scale(circle, self.width / 2, self.height / 2)
            rotated_ellipse = rotate(ellipse, self.angle)
            self._shapely_object = translate(rotated_ellipse, *self.center)
        return self._shapely_object

    @property
    def points(self):
        return np.array(self.shapely_object.exterior.coords)[::-1]

    @property
    def centroid(self):
        return self.center

    @property
    def max_distance(self):
        return np.max([self.width, self.height])

    @property
    def region_measure(self):
        return np.pi * self.width/2 * self.height/2

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

        rad_cc = (xct ** 2 / (self.width / 2.) ** 2) + (yct ** 2 / (self.height / 2.) ** 2)

        inside_indices = np.nonzero(rad_cc < 1.)[0]
        return inside_indices

    def as_artist(self, origin=(0, 0), **kwargs):
        xy = self.center[0] - origin[0], self.center[1] - origin[1]
        return mPatches.Ellipse(xy=xy, width=self.width, height=self.height, angle=self.angle, **kwargs)


class Polygon(Region2D):
    """ closed polygon: ((point1_x, point1_y), (point2_x, point2_y), ..., (point1_x, point1_y)).

    points can be closed or will be closed implicitly.
    polygon is open
    region_specs is legacy
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
            return f'{self.__class__.__name__}({self.points.tolist()})'
        else:
            return f'{self.__class__.__name__}({self.points.tolist()}, {[hole.tolist() for hole in self.holes]})'

    def __str__(self):
        return f'{self.__class__.__name__}(<self.points>, <self.holes>)'

    def __getattr__(self, attr):
        """All non-adapted calls are passed to the object"""
        if attr.startswith('__') and attr.endswith('__'):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self.shapely_object, attr)

    @property
    def points(self):
        return self._points

    @property
    def holes(self):
        return self._holes

    @property
    def region_specs(self):
        return self._region_specs

    @property
    def shapely_object(self):
        if self._shapely_object is None:
            self._shapely_object = shPolygon(self.points, self.holes)
        return self._shapely_object

    @property
    def dimension(self):
        return 2

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
        if np.asarray(points).size == 0:
            return np.array([])
        points_ = asMultiPoint(points)
        prepared_polygon = prep(self.shapely_object)
        mask = list(map(prepared_polygon.contains, points_))
        inside_indices = np.nonzero(mask)[0]
        return inside_indices

    def as_artist(self, **kwargs):
        return mPatches.PathPatch(_polygon_path(self.shapely_object), **kwargs)

    @classmethod
    def from_shapely(cls, polygon):
        if polygon.is_empty:
            return EmptyRegion()
        else:
            points = np.array(polygon.exterior.coords).tolist()
            holes = [np.array(interiors.coords).tolist() for interiors in polygon.interiors]
            return cls(points, holes)


class MultiPolygon(Region2D):

    def __init__(self, polygons):
        self._polygons = polygons
        self._region_specs = None
        self._shapely_object = None

    def __repr__(self):
        return f'{self.__class__.__name__}({self.polygons})'

    def __str__(self):
        return f'{self.__class__.__name__}(<self.polygons>)'

    def __getattr__(self, attr):
        """All non-adapted calls are passed to the object"""
        if attr.startswith('__') and attr.endswith('__'):  # this is needed to enable pickling
            raise AttributeError
        return getattr(self.shapely_object, attr)

    @property
    def points(self):
        return [pol.points for pol in self.polygons]

    @property
    def holes(self):
        return [pol.holes for pol in self.polygons]

    @property
    def polygons(self):
        return self._polygons

    @property
    def region_specs(self):
        return self._region_specs

    @property
    def shapely_object(self):
        if self._shapely_object is None:
            self._shapely_object = shMultiPolygon([pol.shapely_object for pol in self._polygons])
        return self._shapely_object

    @property
    def dimension(self):
        return 2

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
        points_ = asMultiPoint(points)
        prepared_polygon = prep(self.shapely_object)
        mask = list(map(prepared_polygon.contains, points_))
        inside_indices = np.nonzero(mask)[0]
        return inside_indices

    def as_artist(self, **kwargs):
        polygon = shMultiPolygon([pol.shapely_object for pol in self.polygons])
        return mPatches.PathPatch(_polygon_path(polygon), **kwargs)

    @classmethod
    def from_shapely(cls, multipolygon):
        return cls([Polygon.from_shapely(pol) for pol in multipolygon.geoms])


class AxisOrientedCuboid(Region3D):
    """
    3-dimensional convex region with rectangular faces and edges that are parallel to coordinate axes.
    Extension in x-, y-, z-coordinates correspond to length, width, height.
    """
    def __init__(self, corner=(0, 0, 0), length=1, width=1, height=1):
        self._corner = corner
        self._length = length
        self._width = width
        self._height = height

    def __repr__(self):
        return f'{self.__class__.__name__}({tuple(self.corner)}, ' \
               f'{self.length}, {self.width}, {self.height})'

    @classmethod
    def from_intervals(cls, intervals):
        min_x, max_x = intervals[0]
        min_y, max_y = intervals[1]
        min_z, max_z = intervals[2]
        corner = (min_x, min_y, min_z)
        length = max_x - min_x
        width = max_y - min_y
        height = max_z - min_z
        return cls(corner, length, width, height)

    @property
    def corner(self):
        return self._corner

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def dimension(self):
        return 3

    @property
    def points(self):
        return tuple(it.product(*self.intervals))

    @property
    def centroid(self):
        return tuple((cor + dist / 2 for cor, dist in zip(self.corner, (self.length, self.width, self.height))))

    @property
    def bounds(self):
        min_x, min_y, min_z = self.corner
        max_x, max_y, max_z = [cor + dist for cor, dist in zip(self.corner, (self.length, self.width, self.height))]
        return min_x, min_y, min_z, max_x, max_y, max_z

    @property
    def intervals(self):
        """Tuple of ((min_x, max_x), ...)."""
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
        return 2 * (self.length * self.width + self.height * self.width + self.height * self.length)

    def contains(self, points):
        points = np.asarray(points)
        if points.size == 0:
            return np.array([])
        condition_0 = [(points[:, i] >= bound) for i, bound in enumerate(self.bounds[:3])]
        condition_1 = [(points[:, i] < bound) for i, bound in enumerate(self.bounds[3:])]
        condition = np.all(condition_0, axis=0) & np.all(condition_1, axis=0)
        inside_indices = np.nonzero(condition)[0]
        return inside_indices

    def as_artist(self, origin=(0, 0), **kwargs):
        raise NotImplementedError

    def buffer(self, distance, **kwargs):
        mins = self.bounds[:self.dimension]
        maxs = self.bounds[self.dimension:]
        new_mins = [value - distance for value in mins]
        new_maxs = [value + distance for value in maxs]
        return AxisOrientedCuboid(new_mins, *(max_ - min_ for max_, min_ in zip(new_maxs, new_mins)))


# todo: complete implementation
class Cuboid(Region3D):
    """
    3-dimensional convex region with rectangular faces.
    Extension in x-, y-, z-coordinates correspond to length, width, height.
    Corresponding Euler angles are defined by alpha, beta, gamma.
    """
    def __init__(self, corner=(0, 0, 0), length=1, width=1, height=1, alpha=0, beta=0, gamma=0):
        self._corner = corner
        self._length = length
        self._width = width
        self._height = height
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._region_specs = (corner, length, width, height, alpha, beta, gamma)

    def __repr__(self):
        return f'{self.__class__.__name__}({tuple(self.corner)}, ' \
               f'{self.length}, {self.width}, {self.height}, ' \
               f'{self.alpha}, {self.beta}, {self.gamma})'

    @property
    def corner(self):
        return self._corner

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    @property
    def dimension(self):
        return 3

    @property
    def points(self):
        raise NotImplementedError

    @property
    def centroid(self):
        raise NotImplementedError

    @property
    def bounds(self):
        raise NotImplementedError

    @property
    def extent(self):
        raise NotImplementedError

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
        return 2 * (self.length * self.width + self.height * self.width + self.height * self.length)

    def contains(self, points):
        raise NotImplementedError

    def as_artist(self, origin=(0, 0), **kwargs):
        raise NotImplementedError

    def buffer(self, distance, **kwargs):
        raise NotImplementedError


class AxisOrientedHypercuboid(RegionND):
    """
    n-dimensional convex region with edges that are parallel to coordinate axes.
    """
    def __init__(self, corner=(0, 0, 0), lengths=(1, 1, 1)):
        if not len(corner) == len(lengths):
            raise TypeError("corner and lengths must have the same dimension.")
        self._corner = np.array(corner)
        self._lengths = np.array(lengths)

    def __repr__(self):
        return f'{self.__class__.__name__}({tuple(self.corner)}, {tuple(self.lengths)})'

    @classmethod
    def from_intervals(cls, intervals):
        intervals = np.array(intervals)
        corner = intervals[:, 0]
        lengths = np.diff(intervals)[:, 0]
        return cls(corner, lengths)

    @property
    def corner(self):
        return self._corner

    @property
    def lengths(self):
        return self._lengths

    @property
    def dimension(self):
        return len(self.lengths)

    @property
    def intervals(self):
        """Tuple of ((min_x, max_x), ...)."""
        return tuple((lower, upper) for lower, upper in zip(self.bounds[:self.dimension], self.bounds[self.dimension:]))

    @property
    def points(self):
        return tuple(it.product(*self.intervals))

    @property
    def centroid(self):
        return self.corner + self.lengths / 2

    @property
    def bounds(self):
        return np.concatenate([self.corner, self.corner + self.lengths], axis=0)

    @property
    def extent(self):
        return np.abs(self.lengths)

    @property
    def bounding_box(self):
        return self

    @property
    def max_distance(self):
        return np.sqrt(np.sum(self.lengths**2))

    @property
    def region_measure(self):
        return np.product(self.lengths)

    @property
    def subregion_measure(self):
        raise NotImplementedError

    def contains(self, points):
        points = np.asarray(points)
        if points.size == 0:
            return np.array([])
        condition_0 = [(points[:, i] >= bound) for i, bound in enumerate(self.bounds[:self.dimension])]
        condition_1 = [(points[:, i] < bound) for i, bound in enumerate(self.bounds[self.dimension:])]
        condition = np.all(condition_0, axis=0) & np.all(condition_1, axis=0)
        inside_indices = np.nonzero(condition)[0]
        return inside_indices

    def as_artist(self, origin=(0, 0), **kwargs):
        raise NotImplementedError

    def buffer(self, distance):
        mins = self.bounds[:self.dimension] - distance
        maxs = self.bounds[self.dimension:] + distance
        lengths = maxs - mins
        return AxisOrientedHypercuboid(mins, lengths)

    def intersection(self, other):
        """
        Returns a region representing the intersection of this region with
        ``other``.
        """
        raise NotImplementedError

    def symmetric_difference(self, other):
        """
        Returns the union of the two regions minus any areas contained in the
        intersection of the two regions.
        """
        raise NotImplementedError

    def union(self, other):
        """
        Returns a region representing the union of this region with ``other``.
        """
        raise NotImplementedError


def _polygon_path(polygon):
    """
    Constructs a compound matplotlib path from a Shapely geometric object.
    Adapted from https://pypi.org/project/descartes/ (BSD license, copyright Sean Gillies)
    """
    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, 'coords', None) or ob)
        vals = np.ones(n, dtype=mPath.Path.code_type) * mPath.Path.LINETO
        vals[0] = mPath.Path.MOVETO
        return vals

    ptype = polygon.geom_type
    if ptype == 'Polygon':
        polygon = [polygon]
    elif ptype == 'MultiPolygon':
        polygon = [shPolygon(p) for p in polygon]
    else:
        raise ValueError(
            "A polygon or multi-polygon representation is required")

    vertices = np.concatenate([
        np.concatenate([np.asarray(t.exterior)[:, :2]] +
                    [np.asarray(r)[:, :2] for r in t.interiors])
        for t in polygon])
    codes = np.concatenate([
        np.concatenate([coding(t.exterior)] +
                    [coding(r) for r in t.interiors]) for t in polygon])

    return mPath.Path(vertices, codes)
