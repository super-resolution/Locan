"""

Alpha shape

This module provides methods for computing the alpha complex and specific
alpha shapes together with related properties for LocData objects.

Alpha shape is a hull object that
defines a group of localizations bordering a concave hull (which does not have
to be connected and might have holes) [1]_. It depends on a single parameter
`alpha` (here defined to be a distance with a unit equal to the coordinate
units of the given localization data). For `alpha` approaching infinity the
alpha shape is equal to the convex hull.

In this context we call an alpha-complex the subgroup of simplexes of a
Delaunay triangulation that is computed according to Edelsbrunner algorithm.
All localizations that lie on the boundary of the alpha-complex make up the
alpha shape.

Internally we also work with an alpha-independent representation of the
alpha-complex that allows efficient computation of alpha shapes for arbitrary
alpha values.

Simplices are classified as exterior (not part of the alpha complex), interior
(not part of the boundary), regular (part of the boundary but not singular),
and singular (part of the boundary but all simplices of higher dimension they
are incident to are exterior).

References
----------
.. [1] H. Edelsbrunner and E. P. Mücke, Three-dimensional alpha shapes.
   ACM Trans. Graph. 13(1):43-72, 1994.
"""
from __future__ import annotations

import warnings
from collections.abc import Generator, Sequence
from typing import Any, Literal

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay

from locan.data.hulls.alpha_shape_2d import _circumcircle, _half_distance
from locan.data.region import Polygon, Region
from locan.data.region_utils import regions_union

__all__: list[str] = ["AlphaComplex", "AlphaShape"]


def _k_simplex_index_list(
    d: int, k: int
) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
    """
    Given a d-simplex with d being 2 or 3, indexes are provided for all
    k-simplices with k<d that are part of the d-simplex.
    """
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


def _k_simplex_neighbor_index_list(d: int, k: int) -> tuple[int, ...]:
    """
    Given a d-simplex with d being 2 or 3, indexes for neighbor simplexes in
    the scipy.Delaunay triangulation are provided for all k-simplices with k<d
    that are part of the d-simplex.
    """
    if d == 2 and k == 1:
        return 2, 0, 1
    elif d == 3 and k == 2:
        return 3, 2, 1, 0
    else:
        raise ValueError


def _get_k_simplices(
    simplex: Sequence[int], k: int = 1
) -> Generator[list[Any], Any, None]:
    """
    Function to extract k-simplices (e.g. edges) from a d-simplex
    (e.g. triangle; d>k).
    """
    d = len(simplex) - 1
    k_simplices = (
        [simplex[indx] for indx in indices] for indices in _k_simplex_index_list(d, k)  # type: ignore
    )
    return k_simplices


class AlphaComplex:
    """
    Class for an alpha-independent representation of the alpha complex of the
    given points.

    Here the alpha complex is the simplicial subcomplex of the Delaunay
    triangulation together with the intervals defining simplex membership for
    an alpha complex for a specific `alpha`.

    Parameters
    ----------
    points : npt.ArrayLike
        Coordinates of input points with shape (npoints, ndim).
    delaunay : scipy.spatial.Delaunay | None
        Object with attribute `simplices` specifying a list of indices in the
        array of points that define the simplexes in the Delaunay
        triangulation.
        Also, an attribute `neighbor` is required that specifies indices of
        neighboring simplices.
        If None, scipy.stat.Delaunay(points) is computed.

    Attributes
    ----------
    lines : list[tuple[tuple[int, ...], float, float, float]]
        1-simplices (lines) that represent a simplicial subcomplex of the
        Delaunay triangulation with intervals. Array with shape (n_lines, 2).
    triangles : list[tuple[int, float, float, float]]
        2-simplices (triangles) that represent a simplicial subcomplex of the
        Delaunay triangulation with intervals.
        Array with shape (n_triangles, 3).
    tetrahedrons : list[tuple]
        3-simplices (tetrahedrons) that represent a simplicial subcomplex of
        the Delaunay triangulation with intervals.
        Array with shape (n_tetrahedrons, 4).
    dimension : int
        Spatial dimension of the hull.
    delaunay_triangulation : scipy.spatial.Delaunay
        Object with attribute `simplices` specifying a list of indices in the
        array of points that define the simplexes in the Delaunay
        triangulation.
        Also, an attribute `neighbor` is required that specifies indices of
        neighboring simplices.
    """

    def __init__(self, points: npt.ArrayLike, delaunay: Delaunay | None = None) -> None:
        self.lines: list[tuple[tuple[int, ...], float, float, float]]
        self.triangles: list[tuple[int, float, float, float]]
        self.dimension: int | None
        self.delaunay_triangulation: Delaunay | None
        # self.tetrahedrons: list[tuple[int, int, int, int]]  # todo: implement 3d computation

        self.points = np.asarray(points)

        if np.size(self.points) == 0:
            self.dimension = None
            self.delaunay_triangulation = None
            self.lines = []
            self.triangles = []

        elif np.size(self.points) <= 4 and delaunay is None:
            warnings.warn(
                "Not enough points to construct initial simplex (need 4)", stacklevel=1
            )
            self.dimension = None
            self.delaunay_triangulation = None
            self.lines = []
            self.triangles = []

        else:
            self.dimension = np.shape(self.points)[1]
            self.delaunay_triangulation = (
                Delaunay(self.points) if delaunay is None else delaunay
            )

            if self.dimension == 2:
                self.lines, self.triangles = self._compute_2d()
            elif self.dimension == 3:
                self.lines = []
                self.triangles = []
                raise NotImplementedError
            else:
                raise ValueError(
                    f"There is no algorithm available for points with dimension "
                    f"{self.dimension}."
                )

    def _compute_2d(
        self,
    ) -> tuple[
        list[tuple[tuple[int, ...], float, float, float]],
        list[tuple[int, float, float, float]],
    ]:
        """Compute the alpha complex for 2d data."""
        assert self.delaunay_triangulation is not None  # type narrowing # noqa: S101
        n_simplices = len(self.delaunay_triangulation.simplices)

        # circumference for d-simplexes
        circumcircle_radii = np.zeros(n_simplices)
        for n, simplex in enumerate(self.delaunay_triangulation.simplices):
            _, circumcircle_radii[n] = _circumcircle(self.points, simplex)

        alpha_complex_lines: list[tuple[tuple[int, ...], float, float, float]] = []
        alpha_complex_triangles: list[tuple[int, float, float, float]] = []
        for n, (simplex, neighbors, circumcircle_radius) in enumerate(
            zip(
                self.delaunay_triangulation.simplices,
                self.delaunay_triangulation.neighbors,
                circumcircle_radii,
            )
        ):
            interval_a_list = []
            interval_b_list = []
            interval_c_list = []
            for k_simplex, neighbor_index in zip(
                _get_k_simplices(simplex, k=1), _k_simplex_neighbor_index_list(d=2, k=1)
            ):
                neighbor = neighbors[neighbor_index]
                circumcircle_radius_neigh = (
                    np.inf if neighbor == -1 else circumcircle_radii[neighbor]
                )
                interval_b, interval_c = sorted(
                    (circumcircle_radius, circumcircle_radius_neigh)
                )
                interval_a = _half_distance(
                    self.points[k_simplex]
                )  # distance between vertices / 2

                # named tuple Simplices(vertices, interval_a, interval_b, interval_c)
                alpha_complex_lines.append(
                    (tuple(sorted(k_simplex)), interval_a, interval_b, interval_c)  # type: ignore
                )
                # use tuple(sorted()) together with set to get rid of duplicate simplices

                interval_a_list.append(interval_a)
                interval_b_list.append(interval_b)
                interval_c_list.append(interval_c)

            # if alpha is large enough such that ALL k-simplices are singular, regular, or interior,
            # the d-simplex is a singular, regular or interior part of the alpha shape
            alpha_complex_triangles.append(
                (n, max(interval_a_list), max(interval_b_list), max(interval_c_list))  # type: ignore
            )
        alpha_complex_lines = set(alpha_complex_lines)  # type: ignore[assignment]
        alpha_complex_lines = list(alpha_complex_lines)
        return alpha_complex_lines, alpha_complex_triangles

    def get_alpha_complex_lines(
        self,
        alpha: float,
        type: Literal["all", "regular", "singular", "interior", "exterior"] = "all",
    ) -> list[list[int]]:
        """
        Simplicial subcomplex (lines) of the Delaunay triangulation for the
        specific alpha complex for the given `alpha`.

        Parameters
        ----------
        alpha
            Alpha parameter specifying a unique alpha complex.
        type
            Type of alpha complex edges to be included in the graph.
            One of 'all', 'regular', 'singular', 'interior', 'exterior'.

        Returns
        -------
        list[list[int]]
            The indices to specific points in `self.points`.
        """
        if len(self.lines) == 0:
            return []

        if type == "exterior":
            return [list(ac[0]) for ac in self.lines if ac[1] > alpha]
        elif type == "all":
            return [list(ac[0]) for ac in self.lines if ac[1] <= alpha]
        elif type == "singular":
            return [list(ac[0]) for ac in self.lines if ac[1] <= alpha < ac[2]]
        elif type == "regular":
            return [list(ac[0]) for ac in self.lines if ac[2] <= alpha < ac[3]]
        elif type == "interior":
            return [list(ac[0]) for ac in self.lines if ac[3] <= alpha]
        else:
            raise AttributeError(f"Parameter type: {type} is not valid.")
        # a list of lists and not of tuples should be returned since the return value
        # will be used for indexing arrays.

    def get_alpha_complex_triangles(
        self,
        alpha: float,
        type: Literal["all", "regular", "singular", "interior", "exterior"] = "all",
    ) -> list[int]:
        """
        Simplicial subcomplex (triangles) of the Delaunay triangulation for
        the specific alpha complex for the given `alpha`.

        Parameters
        ----------
        alpha
            Alpha parameter specifying a unique alpha complex.
        type
            Type of alpha complex edges to be included in the graph.
            One of 'all', 'regular', 'singular', 'interior', 'exterior'.

        Returns
        -------
        list[int]
            The indices to specific d-simplices in
            `self.delaunay_triangulation.simplices`.
        """
        if len(self.triangles) == 0:
            return []

        if type == "exterior":
            return [ac[0] for ac in self.triangles if ac[1] > alpha]
        elif type == "all":
            return [ac[0] for ac in self.triangles if ac[1] <= alpha]
        elif type == "singular":
            return [ac[0] for ac in self.triangles if ac[1] <= alpha < ac[2]]
        elif type == "regular":
            return [ac[0] for ac in self.triangles if ac[2] <= alpha < ac[3]]
        elif type == "interior":
            return [ac[0] for ac in self.triangles if ac[3] <= alpha]
        else:
            raise AttributeError(f"Parameter type: {type} is not valid.")
        # a list should be returned since the return value
        # will be used for indexing arrays.

    def graph_from_lines(
        self,
        alpha: float,
        type: Literal["all", "regular", "singular", "interior", "exterior"] = "all",
    ) -> nx.Graph:
        """
        Return networkx Graph object with nodes and edges from selected lines.

        Parameters
        ----------
        alpha
            Alpha parameter specifying a unique alpha complex.
        type
            Type of alpha complex edges to be included in the graph.
            One of 'all', 'regular', 'singular', 'interior', 'exterior'.

        Returns
        -------
        networkx.Graph
        """
        G = nx.Graph()

        if len(self.lines) == 0:
            return G

        # positions for all nodes:
        # positions = {i: tuple(point) for i, point in enumerate(self.points)}
        # G.add_nodes_from(positions)
        ac_simplices = self.get_alpha_complex_lines(alpha, type)
        G.add_edges_from(ac_simplices, type=type)
        return G

    def graph_from_triangles(
        self,
        alpha: float,
        type: Literal["all", "regular", "singular", "interior", "exterior"] = "all",
    ) -> nx.Graph:
        """
        Return networkx Graph object with nodes and edges from selected triangles.

        Parameters
        ----------
        alpha
            Alpha parameter specifying a unique alpha complex.
        type
            Type of alpha complex edges to be included in the graph.
            One of 'all', 'regular', 'singular', 'interior', 'exterior'.

        Returns
        -------
        networkx.Graph
        """
        G = nx.MultiGraph()

        if len(self.triangles) == 0:
            return G
        assert self.delaunay_triangulation is not None  # type narrowing # noqa: S101

        # positions for all nodes:
        # positions = {i: tuple(point) for i, point in enumerate(self.points)}
        # G.add_nodes_from(positions)
        triangles = self.get_alpha_complex_triangles(alpha=alpha, type=type)
        for triangle, vertices in zip(
            triangles, self.delaunay_triangulation.simplices[triangles]
        ):
            edges = [vertices[[n, m]] for n, m in [(0, 1), (1, 2), (2, 0)]]
            G.add_edges_from(edges, triangle=triangle)
        return G

    def alpha_shape(self, alpha: float) -> AlphaShape:
        """
        Return the unique alpha shape for `alpha`.

        Parameters
        ----------
        alpha
            Alpha parameter specifying a unique alpha complex.

        Returns
        -------
        AlphaShape
        """
        return AlphaShape(
            points=self.points,
            alpha=alpha,
            alpha_complex=self,
            delaunay=self.delaunay_triangulation,
        )

    def optimal_alpha(self) -> float | None:
        """
        Find the minimum alpha value for which all points belong to the alpha shape,
        in other words, no edges of the Delaunay triangulation are exterior.

        Returns
        -------
        float | None
        """
        if len(self.lines) == 0:
            return_value: float | None = None
        else:
            return_value = np.max([ac[1] for ac in self.lines])
        return return_value

    def alphas(self) -> npt.NDArray[np.float_]:
        """
        Return alpha values at which the corresponding alpha shape changes.

        Returns
        -------
        npt.NDArray[np.float_]
        """
        if len(self.lines) == 0:
            return np.array([], dtype=np.float_)
        else:
            return np.unique([ac[1:] for ac in self.lines])


class AlphaShape:
    """
    Class for the alpha shape of points for a specific alpha value.

    Here the alpha complex is the simplicial subcomplex of the Delaunay
    triangulation for a given alpha value.
    The alpha shape is the union of all simplexes of the alpha complex,
    specified by the boundary points of the alpha complex.

    In order to update an existing AlphaShape object to a different `alpha`
    reset AlphaShape.alpha.

    Parameters
    ----------
    alpha : float
        Alpha parameter specifying a unique alpha complex.
    points : npt.ArrayLike | None
        Coordinates of input points with shape (npoints, ndim).
        Either `points` or `alpha_complex` have to be specified but not both.
    alpha_complex : AlphaComplex | None
        The unfiltered alpha complex with computed interval values.
    delaunay : scipy.spatial.Delaunay | None
        Object with attribute `simplices` specifying a list of indices in the
        array of points that define the simplexes in the Delaunay
        triangulation.
        Also, an attribute `neighbor` is required that specifies indices of
        neighboring simplices.
        If None, scipy.stat.Delaunay(points) is computed.

    Attributes
    ----------
    alpha_complex : AlphaComplex
        The unfiltered alpha complex with computed interval values.
    alpha_shape : npt.NDArray
        The list of k-simplices (edges) from the alpha complex that make up
        the alpha shape.
        Or: Simplicial subcomplex of the Delaunay triangulation with regular
        simplices from the alpha complex.
    region : Region
        Region object.
    connected_components : list[Region]
        Connected components, i.e. a list of the individual unconnected
        polygons that together make up the alpha shape.
    dimension : int
        Spatial dimension of the hull as determined from the dimension of
        `points`
    vertices : npt.NDArray
        Coordinates of points that make up the hull with shape (n_points, 2).
        (regular alpha_shape line-simplices).
    vertex_indices : list[int]
        Indices identifying a polygon of all points that make up the hull
        (regular alpha_shape line-simplices).
    vertices_alpha_shape : npt.NDArray
        Coordinates of points with shape (n_points, 2)
        that make up the interior and boundary of the
        hull (regular, singular and interior alpha_shape line-simplices).
    vertex_alpha_shape_indices : list[int]
        Indices to all points that make up the interior and boundary of the hull.
        (regular, singular and interior alpha_shape line-simplices).
    vertices_connected_components_indices : list[list[int]]
        Indices to the points for each connected component of the alpha shape.
    n_points_on_boundary : float
        The number of points on the hull
        (regular and singular alpha_shape simplices).
    n_points_on_boundary_rel : float
        The number of points on the hull
        (regular and singular alpha_shape simplices)
        relative to all alpha_shape points.
    n_points_alpha_shape : int
        Absolute number of points that are part of the alpha_shape
        (regular, singular and interior alpha_shape simplices).
    n_points_alpha_shape_rel : int
        Absolute number of points that are part of the alpha_shape relative
        to all input points
        (regular, singular and interior alpha_shape simplices).
    region_measure : float
        Hull measure, i.e. area or volume.
    subregion_measure : float
        Measure of the sub-dimensional region, i.e. circumference or surface.
    """

    def __init__(
        self,
        alpha: float,
        points: npt.ArrayLike | None = None,
        alpha_complex: AlphaComplex | None = None,
        delaunay: Delaunay | None = None,
    ) -> None:
        if alpha_complex is None:
            self.alpha_complex: AlphaComplex = AlphaComplex(points, delaunay)  # type: ignore
        else:
            self.alpha_complex = alpha_complex

        self.points = self.alpha_complex.points
        self.dimension = self.alpha_complex.dimension
        self._alpha: float = np.nan
        self._region = None
        self._connected_components: list[Region] | None = None
        self._vertices_connected_components_indices: list[list[int]] | None = None

        self.alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = alpha
        self._region = None
        self._connected_components = None
        self._vertices_connected_components_indices = None

        self.n_points_alpha_shape = self.alpha_complex.graph_from_lines(
            alpha=self.alpha, type="all"
        ).number_of_nodes()
        self.n_points_on_boundary = self.alpha_complex.graph_from_lines(
            alpha=self.alpha, type="regular"
        ).number_of_nodes()
        try:
            self.n_points_on_boundary_rel = (
                self.n_points_on_boundary / self.n_points_alpha_shape
            )
        except ZeroDivisionError:
            self.n_points_on_boundary_rel = float("nan")

        try:
            self.n_points_alpha_shape_rel = self.n_points_alpha_shape / len(self.points)
        except ZeroDivisionError:
            self.n_points_alpha_shape_rel = float("nan")

        self.region_measure = self.region.region_measure
        self.subregion_measure = self.region.subregion_measure

    @property
    def region(self) -> Region:
        if self.dimension == 3:
            raise NotImplementedError(
                "Region for 3D data has not yet been implemented."
            )

        if self._region is None:
            triangles = self.alpha_complex.get_alpha_complex_triangles(alpha=self.alpha)
            if not triangles:
                vertices = []
            else:
                assert (  # type narrowing # noqa: S101
                    self.alpha_complex.delaunay_triangulation is not None
                )
                vertices = self.alpha_complex.delaunay_triangulation.simplices[
                    triangles
                ]
            self._region = regions_union(
                [Polygon(pts) for pts in self.points[vertices]]
            )
        return self._region

    @property
    def connected_components(self) -> list[Region]:
        if self._connected_components is None:
            self._vertices_connected_components_indices = []
            self._connected_components = []
            graph = self.alpha_complex.graph_from_triangles(alpha=self.alpha)
            for cc in nx.connected_components(graph):
                subgraph = graph.subgraph(cc)
                self._vertices_connected_components_indices.append(list(subgraph.nodes))
                triangles = list({edge[2] for edge in subgraph.edges.data("triangle")})
                vertices = self.alpha_complex.delaunay_triangulation.simplices[  # type: ignore
                    triangles
                ]
                region = regions_union([Polygon(pts) for pts in self.points[vertices]])
                self._connected_components.append(region)
        return self._connected_components

    @property
    def vertices_connected_components_indices(self) -> list[list[int]]:
        _ = (
            self.connected_components
        )  # trigger computation of _vertices_connected_components_indices
        return self._vertices_connected_components_indices  # type: ignore

    @property
    def alpha_shape(self) -> list[list[int]]:
        return self.alpha_complex.get_alpha_complex_lines(self.alpha, type="all")

    @property
    def vertices(self) -> npt.NDArray[np.float_]:
        return self.points[self.vertex_indices]

    @property
    def vertex_indices(self) -> list[int]:
        array = self.alpha_complex.get_alpha_complex_lines(self.alpha, type="regular")
        return_value: list[int] = np.unique(array).tolist()
        return return_value

    @property
    def vertices_alpha_shape(self) -> npt.NDArray[np.float_]:
        return self.points[self.vertex_alpha_shape_indices]

    @property
    def vertex_alpha_shape_indices(self) -> list[int]:
        array_r = self.alpha_complex.get_alpha_complex_lines(self.alpha, type="regular")
        array_s = self.alpha_complex.get_alpha_complex_lines(
            self.alpha, type="singular"
        )
        array_i = self.alpha_complex.get_alpha_complex_lines(
            self.alpha, type="interior"
        )
        return_value: list[int] = np.unique(array_r + array_s + array_i).tolist()
        return return_value
