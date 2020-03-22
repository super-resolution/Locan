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
import warnings

import numpy as np
from scipy.spatial import Delaunay
import networkx as nx

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from surepy.data.region import RoiRegion
from surepy.data.hulls.alpha_shape_2d import _circumcircle, _half_distance


__all__ = ['AlphaComplex', 'AlphaShape']


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
    lines : list of tuple with shape (n_lines, 3)
        1-simplices (lines) that represent a simplicial subcomplex of the Delaunay triangulation with intervals.
    triangles : list of tuple with shape (n_triangles, 3)
        2-simplices (triangles) that represent a simplicial subcomplex of the Delaunay triangulation with intervals.
    tetrahedrons : list of tuple with shape (n_tetrahedrons, 3)
        3-simplices (tetrahedrons) that represent a simplicial subcomplex of the Delaunay triangulation with intervals.
    dimension : int
        Spatial dimension of the hull.
    delaunay_triangulation : Delaunay object
        Object with attribute `simplices` specifying a list of indices in the array of points that define the
        simplexes in the Delaunay triangulation.
        Also an attribute `neighbor` is required that specifies indices of neighboring simplices.
    """

    def __init__(self, points, delaunay=None):
        self.points = np.asarray(points)

        if np.size(self.points) == 0:
            self.dimension = None
            self.delaunay_triangulation = None
            self.lines = self.triangles = []

        elif np.size(self.points) <= 4 and delaunay is None:
            warnings.warn("Not enough points to construct initial simplex (need 4)")
            self.dimension = None
            self.delaunay_triangulation = None
            self.lines = self.triangles = []

        else:
            self.dimension = np.shape(self.points)[1]
            self.delaunay_triangulation = Delaunay(self.points) if delaunay is None else delaunay

            if self.dimension == 2:
                self.lines, self.triangles = self._compute_2d()
            elif self.dimension == 3:
                self.lines = self.triangles = []
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

        alpha_complex_lines = []
        alpha_complex_triangles = []
        for n, (simplex, neighbors, circumcircle_radius) in enumerate(zip(self.delaunay_triangulation.simplices,
                                                                     self.delaunay_triangulation.neighbors,
                                                                     circumcircle_radii)):
            interval_a_list = []
            interval_b_list = []
            interval_c_list = []
            for k_simplex, neighbor_index in zip(_get_k_simplices(simplex, k=1),
                                                 _k_simplex_neighbor_index_list(d=2, k=1)):
                neighbor = neighbors[neighbor_index]
                circumcircle_radius_neigh = np.inf if neighbor == -1 else circumcircle_radii[neighbor]
                interval_b, interval_c = sorted((circumcircle_radius, circumcircle_radius_neigh))
                interval_a = _half_distance(self.points[k_simplex])  # distance between vertices / 2

                # named tuple Simplices(vertices, interval_a, interval_b, interval_c)
                alpha_complex_lines.append((tuple(sorted(k_simplex)), interval_a, interval_b, interval_c))
                # use tuple(sorted()) together with set to get rid of duplicate simplices

                interval_a_list.append(interval_a)
                interval_b_list.append(interval_b)
                interval_c_list.append(interval_c)

            # if alpha is large enough such that ALL k-simplices are singular, regular, or interior,
            # the d-simplex is a singular, regular or interior part of the alpha shape
            alpha_complex_triangles.append((n, max(interval_a_list), max(interval_b_list), max(interval_c_list)))
        alpha_complex_lines = set(alpha_complex_lines)
        return alpha_complex_lines, alpha_complex_triangles

    def get_alpha_complex_lines(self, alpha, type='all'):
        """
        Simplicial subcomplex (lines) of the Delaunay triangulation for the specific alpha complex
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
            The indices to specific points in `self.points`.
        """
        if np.size(self.lines) == 0:
            return []

        if type == 'exterior':
            return [list(ac[0]) for ac in self.lines if ac[1] > alpha]
        elif type == 'all':
            return [list(ac[0]) for ac in self.lines if ac[1] <= alpha]
        elif type == 'singular':
            return [list(ac[0]) for ac in self.lines if ac[1] <= alpha < ac[2]]
        elif type == 'regular':
            return [list(ac[0]) for ac in self.lines if ac[2] <= alpha < ac[3]]
        elif type == 'interior':
            return [list(ac[0]) for ac in self.lines if ac[3] <= alpha]
        else:
            raise AttributeError(f'Parameter type: {type} is not valid.')
        # a list of lists and not of tuples should be returned since the return value will be used for indexing arrays.

    def get_alpha_complex_triangles(self, alpha, type='all'):
        """
        Simplicial subcomplex (triangles) of the Delaunay triangulation for the specific alpha complex
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
            The indices to specific d-simplices in `self.delaunay_triangulation.simplices`.
        """
        if np.size(self.triangles) == 0:
            return []

        if type == 'exterior':
            return [ac[0] for ac in self.triangles if ac[1] > alpha]
        elif type == 'all':
            return [ac[0] for ac in self.triangles if ac[1] <= alpha]
        elif type == 'singular':
            return [ac[0] for ac in self.triangles if ac[1] <= alpha < ac[2]]
        elif type == 'regular':
            return [ac[0] for ac in self.triangles if ac[2] <= alpha < ac[3]]
        elif type == 'interior':
            return [ac[0] for ac in self.triangles if ac[3] <= alpha]
        else:
            raise AttributeError(f'Parameter type: {type} is not valid.')
        # a list of lists and not of tuples should be returned since the return value will be used for indexing arrays.

    def graph_from_lines(self, alpha, type='all'):
        """
        Return networkx Graph object with nodes and edges from selected lines.

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

        if np.size(self.lines) == 0:
            return G

        # positions = {i: tuple(point) for i, point in enumerate(self.points)}  # positions for all nodes
        # G.add_nodes_from(positions)
        ac_simplices = self.get_alpha_complex_lines(alpha, type)
        G.add_edges_from(ac_simplices, type=type)
        return G

    def graph_from_triangles(self, alpha, type='all'):
        """
        Return networkx Graph object with nodes and edges from selected triangles.

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
        G = nx.MultiGraph()

        if np.size(self.triangles) == 0:
            return G

        # positions = {i: tuple(point) for i, point in enumerate(self.points)}  # positions for all nodes
        # G.add_nodes_from(positions)
        triangles = self.get_alpha_complex_triangles(alpha=alpha, type=type)
        for triangle, vertices in zip(triangles, self.delaunay_triangulation.simplices[triangles]):
            edges = [vertices[[n, m]] for n, m in [(0, 1), (1, 2), (2, 0)]]
            G.add_edges_from(edges, triangle=triangle)
        return G

    def alpha_shape(self, alpha):
        """
        Return the unique alpha shape for `alpha`.

        Parameters
        ----------
        alpha : float
            Alpha parameter specifying a unique alpha complex.

        Returns
        -------
        float
        """
        return AlphaShape(points=self.points, alpha=alpha,
                          alpha_complex=self, delaunay=self.delaunay_triangulation)

    def optimal_alpha(self):
        """
        Find the minimum alpha value for which all points belong to the alpha shape,
        in other words, no edges of the Delaunay triangulation are exterior.

        Returns
        -------
        float
        """
        if np.size(self.lines) == 0:
            return None
        else:
            return np.max([ac[1] for ac in self.lines])

    def alphas(self):
        """
        Return alpha values at which the corresponding alpha shape changes.

        Returns
        -------
        ndarray
        """
        if np.size(self.lines) == 0:
            return np.array([])
        else:
            return np.unique([ac[1:] for ac in self.lines])


class AlphaShape:
    """
    Class for the alpha shape of points for a specific alpha value.

    Here the alpha complex is the simplicial subcomplex of the Denlaunay triangulation for a given alpha value.
    The alhpa shape is the union of all simplexes of the alpha complex, specified by the boundary points of the
    alpha complex.

    In order to update an existing AlphaShape object to a different `alpha` reset AlphaShape.alpha.

    Parameters
    ----------
    alpha : float
        Alpha parameter specifying a unique alpha complex.
    points : array-like with shape (npoints, ndim) or None
        Coordinates of input points. Either `points` or `alpha_complex` have to be specified but not both.
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
    hull : shapely.geometry.Polygon or shapely.geometry.MultiPolygon or None (in 2D)
        hull object for the alpha shape that contains all d-simplices (triangles in 2D) of the specific alpha complex.
        None if the complex is empty.
    connected_components : list of RoiRegion objects
        Connected components, i.e. a list of the individual unconnected polygons that together make up the alpha shape.
    dimension : int
        Spatial dimension of the hull as determined from the dimension of `points`
    vertices : array of (2,) tuples
        Coordinates of points that make up the hull (regular alpha_shape line-simplices).
    vertex_indices : list of int
        Indices identifying a polygon of all points that make up the hull (regular alpha_shape line-simplices).
    vertices_alpha_shape : array of (2,) tuples
        Coordinates of points that make up the interior and boundary of the hull
        (regular, singular and interior alpha_shape line-simplices).
    vertex_alpha_shape_indices : list of int
        Indices to all points that make up the interior and boundary of the hull.
        (regular, singular and interior alpha_shape line-simplices).
    vertices_connected_components_indices : list of list
        Indices to the points for each connected component of the alpha shape.
    n_points_on_boundary : float
        The number of points on the hull (regular and singular alpha_shape simplices).
    n_points_on_boundary_rel : float
        The number of points on the hull (regular and singular alpha_shape simplices)
        relative to all alpha_shape points.
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

    def __init__(self, alpha, points=None, alpha_complex=None, delaunay=None):
        if alpha_complex is None:
            self.alpha_complex = AlphaComplex(points, delaunay)
        else:
            self.alpha_complex = alpha_complex

        self.points = self.alpha_complex.points
        self.dimension = self.alpha_complex.dimension
        self._alpha = None
        self._hull = None
        self._connected_components = None
        self._vertices_connected_components_indices = None

        self.alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        self._hull = None
        self._connected_components = None
        self._vertices_connected_components_indices = None

        self.n_points_alpha_shape = self.alpha_complex.graph_from_lines(alpha=self.alpha, type='all'). \
            number_of_nodes()
        self.n_points_on_boundary = self.alpha_complex.graph_from_lines(alpha=self.alpha, type='regular'). \
            number_of_nodes()
        try:
            self.n_points_on_boundary_rel = self.n_points_on_boundary / self.n_points_alpha_shape
        except ZeroDivisionError:
            self.n_points_on_boundary_rel = float('nan')

        try:
            self.n_points_alpha_shape_rel = self.n_points_alpha_shape / len(self.points)
        except ZeroDivisionError:
            self.n_points_alpha_shape_rel = float('nan')

        self.region_measure = self.hull.area
        self.subregion_measure = self.hull.length

    @property
    def hull(self):
        if self._hull is None:
            triangles = self.alpha_complex.get_alpha_complex_triangles(alpha=self.alpha)
            if triangles == []:
                vertices = []
            else:
                vertices = self.alpha_complex.delaunay_triangulation.simplices[triangles]
            self._hull = unary_union([Polygon(pts) for pts in self.points[vertices]])
        return self._hull

    @property
    def connected_components(self):
        if self._connected_components is None:
            self._vertices_connected_components_indices = []
            self._connected_components = []
            graph = self.alpha_complex.graph_from_triangles(alpha=self.alpha)
            for cc in nx.connected_components(graph):
                subgraph = graph.subgraph(cc)
                self._vertices_connected_components_indices.append(list(subgraph.nodes))
                triangles = list({edge[2] for edge in subgraph.edges.data('triangle')})
                vertices = self.alpha_complex.delaunay_triangulation.simplices[triangles]
                cc_hull = unary_union([Polygon(pts) for pts in self.points[vertices]])

                if isinstance(cc_hull, MultiPolygon):
                    region = RoiRegion.from_shapely(region_type='shapelyMultiPolygon', shapely_obj=cc_hull)
                elif isinstance(cc_hull, Polygon):
                    region = RoiRegion.from_shapely(region_type='shapelyPolygon', shapely_obj=cc_hull)
                elif cc_hull.is_empty:
                    region = None
                else:
                    raise TypeError('The connected component should be either Polygon or MultiPolygon.')
                self._connected_components.append(region)
        return self._connected_components

    @property
    def vertices_connected_components_indices(self):
        _ = self.connected_components
        return self._vertices_connected_components_indices

    @property
    def alpha_shape(self):
        return self.alpha_complex.get_alpha_complex_lines(self.alpha, type='all')

    @property
    def vertices(self):
        return self.points[self.vertex_indices]

    @property
    def vertex_indices(self):
        array = self.alpha_complex.get_alpha_complex_lines(self.alpha, type='regular')
        return np.unique(array).tolist()

    @property
    def vertices_alpha_shape(self):
        return self.points[self.vertex_alpha_shape_indices]

    @property
    def vertex_alpha_shape_indices(self):
        array_r = self.alpha_complex.get_alpha_complex_lines(self.alpha, type='regular')
        array_s = self.alpha_complex.get_alpha_complex_lines(self.alpha, type='singular')
        array_i = self.alpha_complex.get_alpha_complex_lines(self.alpha, type='interior')
        return np.unique(array_r + array_s + array_i).tolist()

    @property
    def region(self):
        if self.dimension == 3:
            raise NotImplementedError('Region for 3D data has not yet been implemented.')
        else:
            if isinstance(self.hull, MultiPolygon):
                region_ = RoiRegion.from_shapely(region_type='shapelyMultiPolygon', shapely_obj=self.hull)
            elif isinstance(self.hull, Polygon):
                region_ = RoiRegion.from_shapely(region_type='shapelyPolygon', shapely_obj=self.hull)
            elif self.hull.is_empty:
                region_ = None
            else:
                raise TypeError('self.hull should be either Polygon or MultiPolygon.')
            return region_