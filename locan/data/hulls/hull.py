"""

Hull objects of localization data.

This module computes specific hulls for the bounding box, convex hull and oriented bounding box
and related properties for LocData objects.

"""
import numpy as np
import scipy.spatial as spat
from shapely.geometry import MultiPoint as shMultiPoint

from locan.data.region import Rectangle, Polygon


__all__ = ["BoundingBox", "ConvexHull", "OrientedBoundingBox"]


class BoundingBox:
    """
    Class with bounding box computed using numpy operations.

    Parameters
    ----------
    points : numpy.ndarray of double, shape (npoints, ndim)
        Coordinates of input points.

    Attributes
    ----------
    hull : array of arrays
        Array of point coordinates that represent [[min_coordinates], [max_coordinates]].
    dimension : int
        Spatial dimension of hull
    vertices : array of coordinate tuples
        Coordinates of points that make up the hull.
    width : array of float
        Array with differences between max and min for each coordinate.
    region_measure : float
        Hull measure, i.e. area or volume
    subregion_measure : float
        Measure of the sub-dimensional region, i.e. circumference or surface
    region : RoiRegion
        Convert the hull to a RoiRegion object.
    """

    def __init__(self, points):
        if np.size(points) == 0:
            self.dimension = 0
            self.hull = np.array([])
            self.width = np.zeros(self.dimension)
            self.region_measure = 0
            self.subregion_measure = 0
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
            self.region_measure = np.prod(self.width)
            self.subregion_measure = np.sum(self.width) * 2

    @property
    def vertices(self):
        return self.hull.T

    @property
    def region(self):
        if self.dimension == 2:
            region_ = Rectangle(self.hull[0], self.width[0], self.width[1], 0)
            # region_ = RoiRegion(region_type='rectangle', region_specs=(self.hull[0], self.width[0], self.width[1], 0))
        else:
            raise NotImplementedError
        return region_


class _ConvexHullScipy:
    """
    Class with convex hull computed using the scipy.spatial.ConvexHull method.

    Parameters
    ----------
    points : numpy.ndarray of double, shape (npoints, ndim)
        Coordinates of input points.

    Attributes
    ----------
    hull : Hull object
        hull object from the corresponding algorithm
    dimension : int
        spatial dimension of hull
    vertices : array of coordinate tuples
        Coordinates of points that make up the hull.
    vertex_indices : indices for points
        indices identifying a polygon of all points that make up the hull
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

    def __init__(self, points):
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
    def vertices(self):
        return self.hull.points[self.hull.vertices]

    @property
    def region(self):
        if self.dimension > 2:
            raise NotImplementedError(
                "Region for 3D data has not yet been implemented."
            )
        else:
            # closed_vertices = np.append(self.vertices, [self.vertices[0]], axis=0)
            # region_ = RoiRegion(region_type='polygon', region_specs=closed_vertices)
            return Polygon(self.vertices)


class _ConvexHullShapely:
    """
    Class with convex hull computed using the scipy.spatial.ConvexHull method.

    Parameters
    ----------
    points : numpy.ndarray of double, shape (npoints, ndim)
        Coordinates of input points.

    Attributes
    ----------
    hull : Hull object
        Polygon object from the .convex_hull method
    dimension : int
        Spatial dimension of hull
    vertices : array of coordinate tuples
        Coordinates of points that make up the hull.
    vertex_indices : indices for points
        indices identifying a polygon of all points that make up the hull
    points_on_boundary : int
        The absolute number of points on the hull
    points_on_boundary_rel : int
        The number of points on the hull relative to all input points
    region_measure : float
        hull measure, i.e. area or volume
    subregion_measure : float
        measure of the sub-dimensional region, i.e. circumference or surface
    region : RoiRegion
        Convert the hull to a RoiRegion object.
    """

    def __init__(self, points):
        if len(points) < 6:
            unique_points = np.array(list(set(tuple(point) for point in points)))
            if len(unique_points) < 3:
                raise TypeError(
                    "Convex_hull needs at least 3 different points as input."
                )

        self.dimension = np.shape(points)[1]
        if self.dimension >= 3:
            raise TypeError(
                "ConvexHullShapely only takes 1 or 2-dimensional points as input."
            )

        self.hull = shMultiPoint(points).convex_hull
        # todo: set vertex_indices
        # self.vertex_indices = None
        self.points_on_boundary = (
            len(self.hull.exterior.coords) - 1
        )  # the first point is repeated in exterior.coords
        self.points_on_boundary_rel = self.points_on_boundary / len(points)
        self.region_measure = self.hull.area
        self.subregion_measure = self.hull.length

    @property
    def vertices(self):
        return np.array(self.hull.exterior.coords)[:-1]

    @property
    def region(self):
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
    points : numpy.ndarray of double, shape (npoints, ndim)
        Coordinates of input points.
    method : string
        Specific class to compute the convex hull and attributes. One of 'scipy', 'shapely'.

    Attributes
    ----------
    method : string
        Specific class to compute the convex hull and attributes. One of 'scipy', 'shapely'.
    hull : Hull object
        Polygon object from the .convex_hull method
    dimension : int
        Spatial dimension of hull
    vertices : array of coordinate tuples
        Coordinates of points that make up the hull.
    vertex_indices : indices for points
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

    def __init__(self, points, method="scipy"):
        self.method = method
        if method == "scipy":
            self._special_class = _ConvexHullScipy(points)
        elif method == "shapely":
            self._special_class = _ConvexHullShapely(points)
        else:
            raise ValueError(f"The provided method {method} is not available.")

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError

        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._special_class, attr)


class OrientedBoundingBox:
    """
    Class with oriented bounding box computed using the shapely minimum_rotated_rectangle method.

    Parameters
    ----------
    points : numpy.ndarray of double, shape (npoints, ndim)
        Coordinates of input points.

    Attributes
    ----------
    hull : hull object
        Polygon object from the minimum_rotated_rectangle method
    dimension : int
        Spatial dimension of hull
    vertices : array of coordinate tuples
        Coordinates of points that make up the hull.
    width : array of float
        Array with lengths of box edges.
    region_measure : float
        hull measure, i.e. area or volume
    subregion_measure : float
        measure of the sub-dimensional region, i.e. circumference or surface
    region : Region
        Convert the hull to a Region object.
    angle : float
        Orientation defined as angle (in degrees) between the vector from first to last point and x-axis.
    
    """

    def __init__(self, points):
        self.dimension = np.shape(points)[1]

        if self.dimension >= 3:
            raise TypeError(
                "OrientedBoundingBox only takes 1 or 2-dimensional points as input."
            )

        if len(points) < 3:
            self.hull = np.array([])
            self.width = np.zeros(self.dimension)
            self.region_measure = 0
            self.subregion_measure = 0
            self.angle = np.nan
            self.elongation = np.nan
        else:
            self.hull = shMultiPoint(points).minimum_rotated_rectangle
            difference = np.diff(self.vertices[0:3], axis=0)
            self.width = np.array(
                [np.linalg.norm(difference[0]), np.linalg.norm(difference[1])]
            )
            self.region_measure = self.hull.area
            self.subregion_measure = self.hull.length
            self.angle = np.degrees(np.arctan2(difference[0][1], difference[0][0]))
            # numpy.arctan2(y, x) takes reversed x, y arguments.
            self.elongation = 1 - np.divide(*sorted(self.width))

    @property
    def vertices(self):
        return np.array(self.hull.exterior.coords)

    @property
    def region(self):
        if self.dimension == 2:
            # region_ = RoiRegion(region_type='rectangle',
            #                     region_specs=(self.vertices[0], self.width[0], self.width[1], self.angle))
            return Rectangle(self.vertices[0], self.width[0], self.width[1], self.angle)
        else:
            raise NotImplementedError
