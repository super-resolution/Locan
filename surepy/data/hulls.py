import numpy as np

class Hull():
    """
    Abstract class for the hull of a selection

    Parameters
    ----------
    points : array-like of tuples with floats
        coordinates of points for which the hull is computed

    Attributes
    ----------
    hull : hull object
        hull object from the corresponding algorithm
    vertex_indices : indices for points
        indices identifying a polygon of all points that make up the hull
    dimension : int
        spatial dimension of hull
    width : array-like of float
        length in x, y[, and z]-dimension of aligned hulls or max length of oriented hulls
    region_measure : float
        hull measure, i.e. area or volume
    subregion_measure : float
        measure of the sub-dimensional region, i.e. circumference or surface
    """

    def __init__(self, points):
        self.hull = None
        self.vertex_indices = None
        self.dimension = None
        self.width = None
        self.region_measure = None
        self.subregion_measure = None


class Bounding_box():
    """
    Class with bounding box computed using the shapely method.
    """

    def __init__(self, points):
        self.dimension = np.shape(points)[1]
        self.hull = np.array([np.min(points, axis=0), np.max(points, axis=0)])
        self.vertex_indices = None # todo: compute if needed
        self.width = np.diff(self.hull, axis=0).flatten()
        self.region_measure = np.prod(self.width)
        self.subregion_measure = np.sum(self.width)*2


class Convex_hull_scipy():
    """
    Class with convex hull computed using the scipy.spatial.ConvexHull method.

    Parameters
    ----------
    points : ndarray of double, shape (npoints, ndim)
        Coordinates of input points.

    Attributes
    ----------
    hull : hull object
        hull object from the corresponding algorithm
    vertex_indices : indices for points
        indices identifying a polygon of all points that make up the hull
    dimension : int
        spatial dimension of hull
    width : array-like of float
        length in x, y[, and z]-dimension of aligned hulls or max length of oriented hulls
    region_measure : float
        hull measure, i.e. area or volume
    subregion_measure : float
        measure of the sub-dimensional region, i.e. circumference or surface

    """

    def __init__(self, points):
        from scipy.spatial import ConvexHull
        self.dimension = np.shape(points)[1]
        self.hull = ConvexHull(points)
        self.vertex_indices = self.hull.vertices
        self.width = None
        self.region_measure = self.hull.volume if self.dimension==3 else self.hull.area
        self.subregion_measure = None # todo: compute

