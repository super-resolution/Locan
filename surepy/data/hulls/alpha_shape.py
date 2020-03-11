"""

Alpha shape

This module provides methods for computing the alpha complex and specific alpha shapes
together with related properties for LocData objects.

Alpha shape is a hull object that
defines a group of localizations bordering a concave hull (which does not have to be connected and might have
holes) [1]_. It depends on a single parameter `alpha` (here defined to be a distance with a unit equal to the coordinate
units of the given localization data). For `alpha` approaching infinity the alpha shape is equal to the convex hull.

In this context we call an alpha-complex the subgroup of simplexes of a Delaunay triangulation that is computed
according to Edelsbrunner algorithm.  All localizations that lie on the boundary of the alpha-complex make up the
alpha shape.

Internally we also work with an alpha-independent representation of the alpha-complex that allows efficient computation
of alpha shapes for arbitrary alpha values.

Simplices are classified as exterior (not part of the alpha complex), interior (not part of the boundary),
regular (part of the boundary but not singular), and singular (part of the boundary but all simplices of higher
dimension they are incident to are exterior).

References
----------
.. [1] H. Edelsbrunner and E. P. Mücke, Three-dimensional alpha shapes. ACM Trans. Graph. 13(1):43-72, 1994.
"""
import sys
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import networkx as nx

from shapely.geometry.polygon import LinearRing
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

import surepy.data.locdata  # LocData cannot be imported directly due to circular import issues.
from surepy.data.region import RoiRegion


__all__ = ['AlphaComplex', 'AlphaShape']


def _circumcircle(points, simplex):
    """
    Center and radius of circumcircle for one triangle.

    Parameters
    -----------
    points : array of shape (n_points, 2)
        point coordinates
    simplex : list
        list with three indices representing a triangle from three points.

    Returns
    -------
    tuple of float
        Center and radius of circumcircle
    """
    A = np.asarray(points)[simplex]
    M = np.array([np.linalg.norm(A, axis=1)**2, A[:,0], A[:,1], np.ones(3)], dtype=np.float32)
    S = np.array([0.5*np.linalg.det(M[[0,2,3]]), -0.5*np.linalg.det(M[[0,1,3]])])
    a = np.linalg.det(M[1:])
    b = np.linalg.det(M[[0,1,2]])
    return S / a, np.sqrt(b / a + np.linalg.norm(S)**2 / a**2)  # center, radius


def _half_distance(points):
    """
    Half the distance between two points.

    Parameters
    -----------
    points : array of shape (2, 2)
        point coordinates representing a line

    Returns
    -------
    float
        Half the distance between the two points.
    """
    points = np.asarray(points)
    return np.sqrt((points[1, 0] - points[0, 0])**2 + (points[1, 1] - points[0, 1])**2) / 2  # radius = _half_distance
    # this is faster than using: np.linalg.norm(np.diff(A, axis=0)) / 2



def _k_simplex_index_list(d, k):
    """Given a d-simplex with d being 2 or 3, indexes are provided for all k-simplices with k<d
    that are part of the d-simplex."""
    if d == 2:
        if k == 0:
            return 0, 1, 2
        elif k == 1:
            return (0, 1), (1, 2), (2, 0)
        else:
            raise ValueError
    elif d == 3:
        if k == 0:
            return 0, 1, 2, 3
        elif k == 1:
            return (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
        elif k == 2:
            return (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)
        else:
            raise ValueError
    else:
        raise ValueError


def _k_simplex_neighbor_index_list(d, k):
    """Given a d-simplex with d being 2 or 3, indexes for neighbor simplexes in the scipy.Delaunay triangulation
    are provided for all k-simplices with k<d that are part of the d-simplex."""
    if d == 2 and k == 1:
        return 2, 0, 1
    elif d == 3 and k == 2:
        return 3, 2, 1, 0
    else:
        raise ValueError


def _get_k_simplices(simplex, k=1):
    """ Function to extract k-simplices (e.g. edges) from a d-simplex (e.g. triangle; d>k)."""
    d = len(simplex) - 1
    k_simplices = ([simplex[indx] for indx in indices] for indices in _k_simplex_index_list(d, k))
    return k_simplices


class AlphaComplex:
    """
    Class for an alpha-independent representation of the alpha complex of the given points.

    Here the alpha complex is the simplicial subcomplex of the Denlaunay triangulation together with the intervals
    defining simplex membership for an alpha complex for a specific `alpha`.

    Parameters
    ----------
    points : array-like with shape (npoints, ndim)
        Coordinates of input points.
    delaunay : Delaunay object
        Object with attribute `simplices` specifying a list of indices in the array of points that define the
        simplexes in the Delaunay triangulation.
        Also an attribute `neighbor` is required that specifies indices of neighboring simplices.
        If None, scipy.stat.Delaunay(points) is computed.

    Attributes
    ----------
    alpha_complex : list of simplices
        Simplicial subcomplex of the Delaunay triangulation with intervals.
    dimension : int
        Spatial dimension of the hull.
    delaunay_triangulation : scipy.stat.Delaunay obj
        List of indices in the array of points that define simplexes in the Delaunay triangulation.
    """

    def __init__(self, points, delaunay=None):
        self.points = np.asarray(points)
        self.dimension = np.shape(self.points)[1]
        self.delaunay_triangulation = Delaunay(self.points) if delaunay is None else delaunay

        if self.dimension is 2:
            self.alpha_complex = self._compute_2d()
        elif self.dimension is 3:
            self.alpha_complex = None
            raise NotImplementedError
        else:
            raise ValueError(f'There is no algorithm available for points with dimension {self.dimension}.')

    def _compute_2d(self):
        """ Compute the alpha complex for 2d data."""
        n_simplices = len(self.delaunay_triangulation.simplices)

        # circumference for d-simplexes
        circumcircle_radii = np.zeros(n_simplices)
        for n, simplex in enumerate(self.delaunay_triangulation.simplices):
            _, circumcircle_radii[n] = _circumcircle(self.points, simplex)

        alpha_complex = []
        for n, (simplex, neighbors, circumcircle_radius) in enumerate(zip(self.delaunay_triangulation.simplices,
                                                                     self.delaunay_triangulation.neighbors,
                                                                     circumcircle_radii)):
            for k_simplex, neighbor_index in zip(_get_k_simplices(simplex, k=1),
                                                 _k_simplex_neighbor_index_list(d=2, k=1)):
                neighbor = neighbors[neighbor_index]
                circumcircle_radius_neigh = np.inf if neighbor == -1 else circumcircle_radii[neighbor]
                interval_b, interval_c = sorted((circumcircle_radius, circumcircle_radius_neigh))
                interval_a = _half_distance(self.points[k_simplex])  # distance between vertices / 2
                # named tuple Simplices(vertices, interval_a, interval_b, interval_c)
                alpha_complex.append((tuple(sorted(k_simplex)), interval_a, interval_b, interval_c))
                # use tuple(sorted()) together with set to get rid of duplicate simplices
        alpha_complex = set(alpha_complex)
        return alpha_complex

    def get_alpha_complex(self, alpha, type='all'):
        """
        Simplicial subcomplex (edges) of the Delaunay triangulation for the specific alpha complex
        for the given `alpha`.

        Parameters
        ----------
        alpha : float
            Alpha parameter specifying a unique alpha complex.
        type : str
            Type of alpha complex edges to be included in the graph.
            One of 'all', 'regular', 'singular', 'interior', 'exterior'.

        Returns
        -------
        list of list of int
        """
        if type == 'exterior':
            return [list(ac[0]) for ac in self.alpha_complex if ac[1] > alpha]
        elif type == 'all':
            return [list(ac[0]) for ac in self.alpha_complex if ac[1] <= alpha]
        elif type == 'singular':
            return [list(ac[0]) for ac in self.alpha_complex if ac[1] <= alpha < ac[2]]
        elif type == 'regular':
            return [list(ac[0]) for ac in self.alpha_complex if ac[2] <= alpha < ac[3]]
        elif type == 'interior':
            return [list(ac[0]) for ac in self.alpha_complex if ac[3] <= alpha]
        else:
            raise AttributeError(f'Parameter type: {type} is not valid.')
        # a list of lists and not of tuples should be returned since the return value will be used for indexing arrays.

    def to_graph(self, alpha, type='all'):
        """
        Return networkx Graph object.

        Parameters
        ----------
        alpha : float
            Alpha parameter specifying a unique alpha complex.
        type : str
            Type of alpha complex edges to be included in the graph.
            One of 'all', 'regular', 'singular', 'interior', 'exterior'.

        Returns
        -------
        networkx.Graph
        """
        G = nx.Graph()
        # positions = {i: tuple(point) for i, point in enumerate(self.points)}  # positions for all nodes
        # G.add_nodes_from(positions)
        ac_simplices = self.get_alpha_complex(alpha, type)
        G.add_edges_from(ac_simplices, type=type)
        return G

    def alpha_shape(self, alpha):
        return AlphaShape(points=self.points, alpha=alpha,
                          alpha_complex=self, delaunay=self.delaunay_triangulation)

    def optimal_alpha(self):
        raise NotImplementedError

    def alpha_iterator(self):
        raise NotImplementedError


class AlphaShape:
    """
    Class for the alpha shape of points for a specific alpha value.

    Here the alpha complex is the simplicial subcomplex of the Denlaunay triangulation for a given alpha value.
    The alhpa shape is the union of all simplexes of the alpha complex, specified by the boundary points of the
    alpha complex.

    Parameters
    ----------
    points : array-like with shape (npoints, ndim)
        Coordinates of input points.
    alpha : float
        Alpha parameter specifying a unique alpha complex.
    alpha_complex : AlphaComplex or None
        The unfiltered alpha complex with computed interval values.
    delaunay : Delaunay object or None
        Object with attribute `simplices` specifying a list of indices in the array of points that define the
        simplexes in the Delaunay triangulation.
        Also an attribute `neighbor` is required that specifies indices of neighboring simplices.
        If None, scipy.stat.Delaunay(points) is computed.

    Attributes
    ----------
    alpha_complex : AlphaComplex
        The unfiltered alpha complex with computed interval values.
    alpha_shape : ndarray
        The list of k-simplices (edges) from the alpha complex that make up the alpha shape.
        Or: Simplicial subcomplex of the Delaunay triangulation with regular simplices from the alpha complex.
    hull : hull object
        hull object (the alpha shape) representing the boundary points of the alpha complex.
        In 2d: a shapely MultiPolygon object is returned representing a list of unconnected components.
    dimension : int
        Spatial dimension of the hull as determined from the dimension of `points`
    vertices : array of (2,) tuples
        Coordinates of points that make up the hull (regular alpha_shape simplices).
    vertex_indices : list of int
        Indices identifying a polygon of all points that make up the hull (regular alpha_shape simplices).
    vertices_alpha_shape : array of (2,) tuples
        Coordinates of points that make up the interior and boundary of the hull
        (regular, singular and interior alpha_shape simplices).
    vertex_alpha_shape_indices : list of int
        Indices to all points that make up the interior and boundary of the hull.
        (regular, singular and interior alpha_shape simplices).
    vertices_connected_components_indices : list of list
        Indices to the points for each connected component of the alpha shape.
    n_points_on_boundary : float
        The number of points on the hull (regular and singular alpha_shape simplices).
    n_points_on_boundary_rel : float
        The number of points on the hull (regular and singular alpha_shape simplices) relative to all alpha_shape points.
    n_points_alpha_shape : int
        Absolute number of points that are part of the alpha_shape
        (regular, singular and interior alpha_shape simplices).
    n_points_alpha_shape_rel : int
        Absolute number of points that are part of the alpha_shape relative to all input points
        (regular, singular and interior alpha_shape simplices).
    region_measure : float
        Hull measure, i.e. area or volume.
    subregion_measure : float
        Measure of the sub-dimensional region, i.e. circumference or surface.
    region : RoiRegion
        Convert the hull to a RoiRegion object.
    """

    def __init__(self, points, alpha, alpha_complex=None, delaunay=None):
        self.points = np.asarray(points)
        self.dimension = np.shape(self.points)[1]

        if self.dimension != 2:
            raise NotImplementedError

        self.alpha = alpha

        if alpha_complex is None:
            self.alpha_complex = AlphaComplex(points, delaunay)
        else:
            self.alpha_complex = alpha_complex

        self._graph = self.alpha_complex.to_graph(alpha=self.alpha, type='regular')

        self.n_points_alpha_shape = self.alpha_complex.to_graph(alpha=self.alpha, type='all').number_of_nodes()
        self.n_points_on_boundary = self.alpha_complex.to_graph(alpha=self.alpha, type='regular').number_of_nodes()
        self.n_points_on_boundary_rel = self.n_points_on_boundary / self.n_points_alpha_shape
        self.n_points_alpha_shape_rel = self.n_points_alpha_shape / len(self.points)

        self.hull = self._compute_hull()
        self.region_measure = self.hull.area
        self.subregion_measure = self.hull.length

    def _compute_hull(self):
        """Create shapely polygons and define hull."""
        polygons = []
        for cc in nx.connected_components(self._graph):
            subgraph = self._graph.subgraph(cc)
            # order points to create polygon
            indices = [e for item in nx.chain_decomposition(subgraph) for element in item for e in element]
            nodes = []
            last_item = None
            for item in indices:
                if item != last_item:
                    nodes.append(item)
                last_item = item
            polygon = Polygon(self.points[nodes].tolist())
            polygons.append(polygon)

        # find polygon_hole pairs
        mask = []
        for n, m in permutations(range(len(polygons)), 2):
            mask.append(polygons[n].contains(polygons[m]))
        polygon_hole_pairs = np.array(list(permutations(range(len(polygons)), 2)))[mask]

        # check if multi-level cascades exist (polygon inside polygon inside polygon)
        elements = np.ravel(polygon_hole_pairs)
        if len(elements) != len(set(elements)):
            raise NotImplementedError(
                "There are multiple polygons within each other. Dealing with this has not been implemented.")

        # create polygons with holes
        for item in polygon_hole_pairs:
            polygons[item[0]] = Polygon(shell=polygons[item[0]].exterior.coords,
                                        holes=[polygons[item[1]].exterior.coords])
            polygons[item[1]] = None

        return MultiPolygon(polygons)

    @property
    def alpha_shape(self):
        return self.alpha_complex.get_alpha_complex(self.alpha, type='all')

    @property
    def vertices(self):
        return self.points[self.vertex_indices]

    @property
    def vertex_indices(self):
        array = self.alpha_complex.get_alpha_complex(self.alpha, type='regular')
        return np.unique(array).tolist()

    @property
    def vertices_alpha_shape(self):
        return self.points[self.vertex_alpha_shape_indices]

    @property
    def vertex_alpha_shape_indices(self):
        array_r = self.alpha_complex.get_alpha_complex(self.alpha, type='regular')
        array_s = self.alpha_complex.get_alpha_complex(self.alpha, type='singular')
        array_i = self.alpha_complex.get_alpha_complex(self.alpha, type='interior')
        return np.unique(array_r + array_s + array_i).tolist()

    @property
    def vertices_connected_components_indices(self):
        indices_list = []
        for cc in nx.connected_components(self._graph):
            subgraph = self._graph.subgraph(cc)
            indices_list.append(list(subgraph.nodes))
        return indices_list

    @property
    def region(self):
        if self.dimension > 2:
            raise NotImplementedError('Region for 3D data has not yet been implemented.')
        else:
            region_ = RoiRegion.from_shapely(region_type='shapelyMultiPolygon', shapely_obj=self.hull)
            return region_
