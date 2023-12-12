import math

import matplotlib.pyplot as plt  # needed for visualization
import matplotlib.tri as mtri  # needed for visualization
import networkx as nx
import numpy as np
import pytest

from locan import AlphaComplex, AlphaShape, EmptyRegion, Region2D


@pytest.mark.visual
def test_AlphaComplex_visual(locdata_2d):
    points = locdata_2d.coordinates
    alpha_complex = AlphaComplex(points)

    alpha = 1.6
    ac_simplices_all = alpha_complex.get_alpha_complex_lines(alpha, type="all")
    ac_simplices_exterior = alpha_complex.get_alpha_complex_lines(
        alpha, type="exterior"
    )
    ac_simplices_interior = alpha_complex.get_alpha_complex_lines(
        alpha, type="interior"
    )
    ac_simplices_regular = alpha_complex.get_alpha_complex_lines(alpha, type="regular")
    ac_simplices_singular = alpha_complex.get_alpha_complex_lines(
        alpha, type="singular"
    )

    lengths = [
        len(sim)
        for sim in [
            ac_simplices_all,
            ac_simplices_exterior,
            ac_simplices_interior,
            ac_simplices_regular,
            ac_simplices_singular,
        ]
    ]
    assert lengths == [8, 2, 0, 6, 2]

    locdata_2d.data.plot(*locdata_2d.coordinate_properties, kind="scatter")
    for simp in ac_simplices_all:
        plt.plot(points[simp, 0], points[simp, 1], "-b")
    for simp in ac_simplices_interior:
        plt.plot(points[simp, 0], points[simp, 1], "--g")
    for simp in ac_simplices_regular:
        plt.plot(points[simp, 0], points[simp, 1], "--r")
    for simp in ac_simplices_singular:
        plt.plot(points[simp, 0], points[simp, 1], "--y")
    plt.show()


@pytest.mark.visual
def test_AlphaComplex_1_visual():
    points = np.array(
        [
            [10, 10],
            [10, 14],
            [11, 11],
            [12, 11],
            [13, 10],
            [13, 13],
            [15, 14],
            [17, 16],
            [10, 24],
        ]
    )

    alpha_complex = AlphaComplex(points)

    alpha = 2.2
    ac_simplices_all = alpha_complex.get_alpha_complex_lines(alpha, type="all")
    ac_simplices_exterior = alpha_complex.get_alpha_complex_lines(  # noqa: F841
        alpha, type="exterior"
    )
    ac_simplices_interior = alpha_complex.get_alpha_complex_lines(
        alpha, type="interior"
    )
    ac_simplices_regular = alpha_complex.get_alpha_complex_lines(alpha, type="regular")
    ac_simplices_singular = alpha_complex.get_alpha_complex_lines(
        alpha, type="singular"
    )

    H = alpha_complex.graph_from_lines(alpha, type="regular")

    fig, ax = plt.subplots(1, 3)
    nx.draw_networkx(H, points, with_labels=True, ax=ax[0])
    # plt.show()

    plt.scatter(*points.T)
    for simp in ac_simplices_all:
        ax[1].plot(points[simp, 0], points[simp, 1], "-b")
    for simp in ac_simplices_interior:
        ax[1].plot(points[simp, 0], points[simp, 1], "--g")
    for simp in ac_simplices_regular:
        ax[1].plot(points[simp, 0], points[simp, 1], "--r")
    for simp in ac_simplices_singular:
        ax[1].plot(points[simp, 0], points[simp, 1], "--y")
    # plt.show()

    triangles_all = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="all")
    ]
    triangles_exterior = alpha_complex.delaunay_triangulation.simplices[  # noqa: F841
        alpha_complex.get_alpha_complex_triangles(alpha, type="exterior")
    ]
    triangles_interior = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="interior")
    ]
    triangles_regular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="regular")
    ]
    triangles_singular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="singular")
    ]

    tri = mtri.Triangulation(*points.T, triangles=triangles_all)
    ax[2].triplot(tri, "-b")
    tri = mtri.Triangulation(*points.T, triangles=triangles_interior)
    ax[2].triplot(tri, "--g")
    tri = mtri.Triangulation(*points.T, triangles=triangles_regular)
    ax[2].triplot(tri, "--r")
    tri = mtri.Triangulation(*points.T, triangles=triangles_singular)
    ax[2].triplot(tri, "--y")
    plt.show()


@pytest.mark.visual
def test_AlphaComplex_2_visual():
    points = np.array(
        [
            [10, 10],
            [10, 14],
            [11, 11],
            [12, 11],
            [13, 10],
            [13, 13],
            [15, 14],
            [17, 16],
            [20, 20],
            [20, 24],
            [21, 21],
            [22, 21],
            [23, 20],
            [23, 23],
            [25, 24],
            [10, 24],
        ]
    )

    alpha_complex = AlphaComplex(points)

    alpha = 2.2
    ac_simplices_all = alpha_complex.get_alpha_complex_lines(alpha, type="all")
    ac_simplices_exterior = alpha_complex.get_alpha_complex_lines(  # noqa: F841
        alpha, type="exterior"
    )
    ac_simplices_interior = alpha_complex.get_alpha_complex_lines(
        alpha, type="interior"
    )
    ac_simplices_regular = alpha_complex.get_alpha_complex_lines(alpha, type="regular")
    ac_simplices_singular = alpha_complex.get_alpha_complex_lines(
        alpha, type="singular"
    )

    H = alpha_complex.graph_from_lines(alpha, type="regular")

    fig, ax = plt.subplots(1, 3)
    nx.draw_networkx(H, points, with_labels=True, ax=ax[0])
    # plt.show()

    plt.scatter(*points.T)
    for simp in ac_simplices_all:
        ax[1].plot(points[simp, 0], points[simp, 1], "-b")
    for simp in ac_simplices_interior:
        ax[1].plot(points[simp, 0], points[simp, 1], "--g")
    for simp in ac_simplices_regular:
        ax[1].plot(points[simp, 0], points[simp, 1], "--r")
    for simp in ac_simplices_singular:
        ax[1].plot(points[simp, 0], points[simp, 1], "--y")
    # plt.show()

    triangles_all = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="all")
    ]
    triangles_exterior = alpha_complex.delaunay_triangulation.simplices[  # noqa: F841
        alpha_complex.get_alpha_complex_triangles(alpha, type="exterior")
    ]
    triangles_interior = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="interior")
    ]
    triangles_regular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="regular")
    ]
    triangles_singular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="singular")
    ]

    tri = mtri.Triangulation(*points.T, triangles=triangles_all)
    ax[2].triplot(tri, "-b")
    tri = mtri.Triangulation(*points.T, triangles=triangles_interior)
    ax[2].triplot(tri, "--g")
    tri = mtri.Triangulation(*points.T, triangles=triangles_regular)
    ax[2].triplot(tri, "--r")
    tri = mtri.Triangulation(*points.T, triangles=triangles_singular)
    ax[2].triplot(tri, "--y")
    plt.show()


@pytest.mark.visual
def test_AlphaComplex_3_visual():
    points = np.array(
        [
            [10, 10],
            [10, 20],
            [20, 20],
            [20, 10],
            [10, 15],
            [15, 20],
            [20, 15],
            [15, 10],
            [10, 12],
            [10, 18],
            [20, 18],
            [20, 12],
            [17, 20],
            [12, 10],
            [17, 10],
            [12, 20],
            [11, 11],
            [11, 19],
            [19, 19],
            [19, 11],
            [11, 15],
            [15, 19],
            [19, 15],
            [15, 11],
            [13, 11],
            [11.5, 12],
            [12, 18],
            [13, 19],
            [17, 19],
            [18, 18],
            [17, 11],
            [18, 12],
            [15, 15],
        ]
    )

    alpha_complex = AlphaComplex(points)

    alpha = 2.2
    ac_simplices_all = alpha_complex.get_alpha_complex_lines(alpha, type="all")
    ac_simplices_exterior = alpha_complex.get_alpha_complex_lines(  # noqa: F841
        alpha, type="exterior"
    )
    ac_simplices_interior = alpha_complex.get_alpha_complex_lines(
        alpha, type="interior"
    )
    ac_simplices_regular = alpha_complex.get_alpha_complex_lines(alpha, type="regular")
    ac_simplices_singular = alpha_complex.get_alpha_complex_lines(
        alpha, type="singular"
    )

    H = alpha_complex.graph_from_lines(alpha, type="regular")

    fig, ax = plt.subplots(1, 3)
    nx.draw_networkx(H, points, with_labels=True, ax=ax[0])
    # plt.show()

    plt.scatter(*points.T)
    for simp in ac_simplices_all:
        ax[1].plot(points[simp, 0], points[simp, 1], "-b")
    for simp in ac_simplices_interior:
        ax[1].plot(points[simp, 0], points[simp, 1], "--g")
    for simp in ac_simplices_regular:
        ax[1].plot(points[simp, 0], points[simp, 1], "--r")
    for simp in ac_simplices_singular:
        ax[1].plot(points[simp, 0], points[simp, 1], "--y")
    # plt.show()

    triangles_all = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="all")
    ]
    triangles_exterior = alpha_complex.delaunay_triangulation.simplices[  # noqa: F841
        alpha_complex.get_alpha_complex_triangles(alpha, type="exterior")
    ]
    triangles_interior = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="interior")
    ]
    triangles_regular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="regular")
    ]
    triangles_singular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="singular")
    ]

    tri = mtri.Triangulation(*points.T, triangles=triangles_all)
    ax[2].triplot(tri, "-b")
    tri = mtri.Triangulation(*points.T, triangles=triangles_interior)
    ax[2].triplot(tri, "--g")
    tri = mtri.Triangulation(*points.T, triangles=triangles_regular)
    ax[2].triplot(tri, "--r")
    tri = mtri.Triangulation(*points.T, triangles=triangles_singular)
    ax[2].triplot(tri, "--y")
    plt.show()


def test_AlphaComplex_0():
    points = np.array([])
    assert np.size(points) == 0
    alpha_complex = AlphaComplex(points)
    assert len(alpha_complex.triangles) == 0
    assert len(alpha_complex.lines) == 0
    assert alpha_complex.dimension is None
    assert (
        alpha_complex.get_alpha_complex_lines(
            alpha=alpha_complex.optimal_alpha(), type="exterior"
        )
        == []
    )
    assert alpha_complex.optimal_alpha() is None
    assert alpha_complex.alphas().size == 0
    # assert isinstance(alpha_complex.alpha_shape(alpha=2.2), AlphaShape)

    alpha = 2.2
    ac_simplices_all = alpha_complex.get_alpha_complex_lines(alpha, type="all")
    ac_simplices_exterior = alpha_complex.get_alpha_complex_lines(
        alpha, type="exterior"
    )
    ac_simplices_interior = alpha_complex.get_alpha_complex_lines(
        alpha, type="interior"
    )
    ac_simplices_regular = alpha_complex.get_alpha_complex_lines(alpha, type="regular")
    ac_simplices_singular = alpha_complex.get_alpha_complex_lines(
        alpha, type="singular"
    )

    lengths = [
        len(sim)
        for sim in [
            ac_simplices_all,
            ac_simplices_exterior,
            ac_simplices_interior,
            ac_simplices_regular,
            ac_simplices_singular,
        ]
    ]
    assert lengths == [0, 0, 0, 0, 0]

    triangles_all = alpha_complex.get_alpha_complex_triangles(alpha, type="all")

    assert len(triangles_all) == 0

    H = alpha_complex.graph_from_lines(alpha, type="regular")
    assert isinstance(H, nx.Graph)
    H = alpha_complex.graph_from_triangles(alpha, type="regular")
    assert isinstance(H, nx.Graph)


def test_AlphaComplex_1():
    points = np.array(
        [
            [10, 10],
            [10, 14],
            [11, 11],
            [12, 11],
            [13, 10],
            [13, 13],
            [15, 14],
            [17, 16],
            [10, 24],
        ]
    )
    assert len(points) == 9
    alpha_complex = AlphaComplex(points)
    assert len(alpha_complex.triangles) == 11
    assert len(alpha_complex.lines) == 19
    assert alpha_complex.dimension == 2
    assert (
        alpha_complex.get_alpha_complex_lines(
            alpha=alpha_complex.optimal_alpha(), type="exterior"
        )
        == []
    )
    assert alpha_complex.optimal_alpha() in alpha_complex.alphas()
    assert np.max(alpha_complex.alphas()) == np.inf
    assert isinstance(alpha_complex.alpha_shape(alpha=2.2), AlphaShape)

    alpha = 2.2
    ac_simplices_all = alpha_complex.get_alpha_complex_lines(alpha, type="all")
    ac_simplices_exterior = alpha_complex.get_alpha_complex_lines(
        alpha, type="exterior"
    )
    ac_simplices_interior = alpha_complex.get_alpha_complex_lines(
        alpha, type="interior"
    )
    ac_simplices_regular = alpha_complex.get_alpha_complex_lines(alpha, type="regular")
    ac_simplices_singular = alpha_complex.get_alpha_complex_lines(
        alpha, type="singular"
    )

    lengths = [
        len(sim)
        for sim in [
            ac_simplices_all,
            ac_simplices_exterior,
            ac_simplices_interior,
            ac_simplices_regular,
            ac_simplices_singular,
        ]
    ]
    assert lengths == [13, 6, 5, 5, 3]

    triangles_all = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="all")
    ]
    triangles_exterior = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="exterior")
    ]
    triangles_interior = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="interior")
    ]
    triangles_regular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="regular")
    ]
    triangles_singular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="singular")
    ]

    lengths = [
        len(sim)
        for sim in [
            triangles_all,
            triangles_exterior,
            triangles_interior,
            triangles_regular,
            triangles_singular,
        ]
    ]
    assert lengths == [6, 5, 1, 4, 1]

    H = alpha_complex.graph_from_lines(alpha, type="regular")
    assert isinstance(H, nx.Graph)
    H = alpha_complex.graph_from_triangles(alpha, type="regular")
    assert isinstance(H, nx.Graph)

    alpha = 1
    ac_simplices_all = alpha_complex.get_alpha_complex_lines(alpha, type="all")
    ac_simplices_exterior = alpha_complex.get_alpha_complex_lines(
        alpha, type="exterior"
    )
    ac_simplices_interior = alpha_complex.get_alpha_complex_lines(
        alpha, type="interior"
    )
    ac_simplices_regular = alpha_complex.get_alpha_complex_lines(alpha, type="regular")
    ac_simplices_singular = alpha_complex.get_alpha_complex_lines(
        alpha, type="singular"
    )

    lengths = [
        len(sim)
        for sim in [
            ac_simplices_all,
            ac_simplices_exterior,
            ac_simplices_interior,
            ac_simplices_regular,
            ac_simplices_singular,
        ]
    ]
    assert lengths == [3, 16, 0, 0, 3]

    triangles_all = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="all")
    ]
    triangles_exterior = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="exterior")
    ]
    triangles_interior = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="interior")
    ]
    triangles_regular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="regular")
    ]
    triangles_singular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="singular")
    ]

    lengths = [
        len(sim)
        for sim in [
            triangles_all,
            triangles_exterior,
            triangles_interior,
            triangles_regular,
            triangles_singular,
        ]
    ]
    assert lengths == [0, 11, 0, 0, 0]

    H = alpha_complex.graph_from_lines(alpha, type="all")
    assert isinstance(H, nx.Graph)
    H = alpha_complex.graph_from_triangles(alpha, type="regular")
    assert isinstance(H, nx.Graph)


def test_AlphaComplex_2():
    points = np.array(
        [
            [10, 10],
            [10, 14],
            [11, 11],
            [12, 11],
            [13, 10],
            [13, 13],
            [15, 14],
            [17, 16],
            [20, 20],
            [20, 24],
            [21, 21],
            [22, 21],
            [23, 20],
            [23, 23],
            [25, 24],
            [10, 24],
        ]
    )
    assert len(points) == 16
    alpha_complex = AlphaComplex(points)
    assert len(alpha_complex.lines) == 38
    assert alpha_complex.dimension == 2
    assert (
        alpha_complex.get_alpha_complex_lines(
            alpha=alpha_complex.optimal_alpha(), type="exterior"
        )
        == []
    )
    assert alpha_complex.optimal_alpha() in alpha_complex.alphas()
    assert np.max(alpha_complex.alphas()) == np.inf
    assert isinstance(alpha_complex.alpha_shape(alpha=2.2), AlphaShape)

    alpha = 2.2
    ac_simplices_all = alpha_complex.get_alpha_complex_lines(alpha, type="all")
    ac_simplices_exterior = alpha_complex.get_alpha_complex_lines(
        alpha, type="exterior"
    )
    ac_simplices_interior = alpha_complex.get_alpha_complex_lines(
        alpha, type="interior"
    )
    ac_simplices_regular = alpha_complex.get_alpha_complex_lines(alpha, type="regular")
    ac_simplices_singular = alpha_complex.get_alpha_complex_lines(
        alpha, type="singular"
    )

    lengths = [
        len(sim)
        for sim in [
            ac_simplices_all,
            ac_simplices_exterior,
            ac_simplices_interior,
            ac_simplices_regular,
            ac_simplices_singular,
        ]
    ]
    assert lengths == [25, 13, 10, 10, 5]

    triangles_all = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="all")
    ]
    triangles_exterior = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="exterior")
    ]
    triangles_interior = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="interior")
    ]
    triangles_regular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="regular")
    ]
    triangles_singular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="singular")
    ]

    lengths = [
        len(sim)
        for sim in [
            triangles_all,
            triangles_exterior,
            triangles_interior,
            triangles_regular,
            triangles_singular,
        ]
    ]
    assert lengths == [12, 11, 3, 7, 2]

    H = alpha_complex.graph_from_lines(alpha, type="regular")
    assert isinstance(H, nx.Graph)
    H = alpha_complex.graph_from_triangles(alpha, type="regular")
    assert isinstance(H, nx.Graph)


def test_AlphaComplex_3():
    points = np.array(
        [
            [10, 10],
            [10, 20],
            [20, 20],
            [20, 10],
            [10, 15],
            [15, 20],
            [20, 15],
            [15, 10],
            [10, 12],
            [10, 18],
            [20, 18],
            [20, 12],
            [17, 20],
            [12, 10],
            [17, 10],
            [12, 20],
            [11, 11],
            [11, 19],
            [19, 19],
            [19, 11],
            [11, 15],
            [15, 19],
            [19, 15],
            [15, 11],
            [13, 11],
            [11.5, 12],
            [12, 18],
            [13, 19],
            [17, 19],
            [18, 18],
            [17, 11],
            [18, 12],
            [15, 15],
        ]
    )
    assert len(points) == 33
    alpha_complex = AlphaComplex(points)
    assert len(alpha_complex.lines) == 80
    assert alpha_complex.dimension == 2
    assert (
        alpha_complex.get_alpha_complex_lines(
            alpha=alpha_complex.optimal_alpha(), type="exterior"
        )
        == []
    )
    assert alpha_complex.optimal_alpha() in alpha_complex.alphas()
    assert np.max(alpha_complex.alphas()) == np.inf
    assert isinstance(alpha_complex.alpha_shape(alpha=2.2), AlphaShape)

    alpha = 2.2
    ac_simplices_all = alpha_complex.get_alpha_complex_lines(alpha, type="all")
    ac_simplices_exterior = alpha_complex.get_alpha_complex_lines(
        alpha, type="exterior"
    )
    ac_simplices_interior = alpha_complex.get_alpha_complex_lines(
        alpha, type="interior"
    )
    ac_simplices_regular = alpha_complex.get_alpha_complex_lines(alpha, type="regular")
    ac_simplices_singular = alpha_complex.get_alpha_complex_lines(
        alpha, type="singular"
    )

    lengths = [
        len(sim)
        for sim in [
            ac_simplices_all,
            ac_simplices_exterior,
            ac_simplices_interior,
            ac_simplices_regular,
            ac_simplices_singular,
        ]
    ]
    assert lengths == [75, 5, 40, 28, 7]

    triangles_all = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="all")
    ]
    triangles_exterior = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="exterior")
    ]
    triangles_interior = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="interior")
    ]
    triangles_regular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="regular")
    ]
    triangles_singular = alpha_complex.delaunay_triangulation.simplices[
        alpha_complex.get_alpha_complex_triangles(alpha, type="singular")
    ]

    lengths = [
        len(sim)
        for sim in [
            triangles_all,
            triangles_exterior,
            triangles_interior,
            triangles_regular,
            triangles_singular,
        ]
    ]
    assert lengths == [39, 9, 8, 28, 3]

    H = alpha_complex.graph_from_lines(alpha=2.2, type="regular")
    assert isinstance(H, nx.Graph)
    H = alpha_complex.graph_from_triangles(alpha, type="regular")
    assert isinstance(H, nx.Graph)


def test_AlphaShape_0():
    points = np.array([])

    alpha = 2.2
    alpha_shape = AlphaShape(alpha, points=points)
    assert len(alpha_shape.alpha_shape) == 0
    assert isinstance(alpha_shape.alpha_shape, list)
    assert alpha_shape.dimension is None
    assert isinstance(alpha_shape.region, EmptyRegion)
    assert alpha_shape.region_measure == 0
    assert alpha_shape.subregion_measure == 0
    assert alpha_shape.n_points_alpha_shape == 0
    assert math.isnan(alpha_shape.n_points_alpha_shape_rel)
    assert alpha_shape.n_points_on_boundary == 0
    assert math.isnan(alpha_shape.n_points_on_boundary_rel)
    assert alpha_shape.vertex_indices == []
    assert np.size(alpha_shape.vertices) == 0
    assert np.size(alpha_shape.vertices_alpha_shape) == 0
    assert np.size(alpha_shape.vertex_alpha_shape_indices) == 0

    cc = alpha_shape.connected_components
    assert len(cc) == 0
    assert alpha_shape.vertices_connected_components_indices == []


def test_AlphaShape_1():
    points = np.array(
        [
            [10, 10],
            [10, 14],
            [11, 11],
            [12, 11],
            [13, 10],
            [13, 13],
            [15, 14],
            [17, 16],
            [10, 24],
        ]
    )
    assert len(points) == 9

    alpha = 2.2
    alpha_shape = AlphaShape(alpha, points=points)
    assert len(alpha_shape.alpha_shape) == 13
    assert isinstance(alpha_shape.alpha_shape, list)
    assert alpha_shape.dimension == 2
    assert isinstance(alpha_shape.region, Region2D)
    assert alpha_shape.region_measure == 10.5
    assert alpha_shape.subregion_measure == 13.16227766016838
    assert alpha_shape.n_points_alpha_shape == 8
    assert alpha_shape.n_points_alpha_shape_rel == 0.8888888888888888
    assert alpha_shape.n_points_on_boundary == 5
    assert alpha_shape.n_points_on_boundary_rel == 0.625
    assert np.array_equal(alpha_shape.vertex_indices, [0, 1, 2, 4, 5])
    assert alpha_shape.vertices.shape == (5, 2)
    assert alpha_shape.vertices_alpha_shape.shape == (8, 2)
    assert len(alpha_shape.vertex_alpha_shape_indices) == 8
    cc = alpha_shape.connected_components
    assert len(cc) == 1
    assert isinstance(cc[0], Region2D)
    assert alpha_shape.vertices_connected_components_indices == [[2, 1, 0, 5, 3, 4]]

    alpha = 1
    # alpha_shape = AlphaShape(alpha, points=points)
    alpha_shape.alpha = 1
    assert len(alpha_shape.alpha_shape) == 3
    assert isinstance(alpha_shape.alpha_shape, list)
    assert alpha_shape.dimension == 2
    assert isinstance(alpha_shape.region, EmptyRegion)
    assert alpha_shape.region_measure == 0
    assert alpha_shape.subregion_measure == 0
    assert alpha_shape.n_points_alpha_shape == 4
    assert alpha_shape.n_points_alpha_shape_rel == 0.4444444444444444
    assert alpha_shape.n_points_on_boundary == 0
    assert alpha_shape.n_points_on_boundary_rel == 0
    assert np.array_equal(alpha_shape.vertex_indices, [])
    assert alpha_shape.vertices.shape == (0, 2)
    assert alpha_shape.vertices_alpha_shape.shape == (4, 2)
    assert len(alpha_shape.vertex_alpha_shape_indices) == 4
    cc = alpha_shape.connected_components
    assert len(cc) == 0
    assert alpha_shape.vertices_connected_components_indices == []


@pytest.mark.visual
def test_AlphaShape_2_visual():
    points = np.array(
        [
            [10, 10],
            [10, 14],
            [11, 11],
            [12, 11],
            [13, 10],
            [13, 13],
            [15, 14],
            [17, 16],
            [20, 20],
            [20, 24],
            [21, 21],
            [22, 21],
            [23, 20],
            [23, 23],
            [25, 24],
            [10, 24],
        ]
    )
    assert len(points) == 16

    alpha = 2.2
    alpha_shape = AlphaShape(alpha, points=points)
    H = alpha_shape.alpha_complex.graph_from_lines(alpha, type="regular")
    assert isinstance(H, nx.Graph)
    # visualization
    nx.draw_networkx(H, points, with_labels=True)
    plt.show()


def test_AlphaShape_2():
    points = np.array(
        [
            [10, 10],
            [10, 14],
            [11, 11],
            [12, 11],
            [13, 10],
            [13, 13],
            [15, 14],
            [17, 16],
            [20, 20],
            [20, 24],
            [21, 21],
            [22, 21],
            [23, 20],
            [23, 23],
            [25, 24],
            [10, 24],
        ]
    )
    assert len(points) == 16

    alpha = 2.2
    alpha_shape = AlphaShape(alpha, points=points)
    H = alpha_shape.alpha_complex.graph_from_lines(alpha, type="regular")
    assert isinstance(H, nx.Graph)

    assert len(alpha_shape.alpha_shape) == 25
    assert isinstance(alpha_shape.alpha_shape, list)
    assert alpha_shape.dimension == 2
    assert isinstance(alpha_shape.region, Region2D)
    assert alpha_shape.region_measure == 21.0
    assert alpha_shape.subregion_measure == 26.32455532033676
    assert alpha_shape.n_points_alpha_shape == 15
    assert alpha_shape.n_points_alpha_shape_rel == 0.9375
    assert alpha_shape.n_points_on_boundary == 10
    assert alpha_shape.n_points_on_boundary_rel == 0.6666666666666666
    assert np.array_equal(alpha_shape.vertex_indices, [0, 1, 2, 4, 5, 8, 9, 10, 12, 13])
    assert alpha_shape.vertices.shape == (10, 2)
    assert alpha_shape.vertices_alpha_shape.shape == (15, 2)
    assert len(alpha_shape.vertex_alpha_shape_indices) == 15
    cc = alpha_shape.connected_components
    assert len(cc) == 2
    assert isinstance(cc[0], Region2D)
    assert alpha_shape.vertices_connected_components_indices == [
        [13, 11, 12, 8, 10, 9],
        [2, 1, 0, 5, 3, 4],
    ]


def test_AlphaShape_3():
    points = np.array(
        [
            [10, 10],
            [10, 20],
            [20, 20],
            [20, 10],
            [10, 15],
            [15, 20],
            [20, 15],
            [15, 10],
            [10, 12],
            [10, 18],
            [20, 18],
            [20, 12],
            [17, 20],
            [12, 10],
            [17, 10],
            [12, 20],
            [11, 11],
            [11, 19],
            [19, 19],
            [19, 11],
            [11, 15],
            [15, 19],
            [19, 15],
            [15, 11],
            [13, 11],
            [11.5, 12],
            [12, 18],
            [13, 19],
            [17, 19],
            [18, 18],
            [17, 11],
            [18, 12],
            [15, 15],
        ]
    )
    assert len(points) == 33

    alpha = 2.2
    alpha_shape = AlphaShape(alpha, points=points)
    H = alpha_shape.alpha_complex.graph_from_lines(alpha, type="regular")
    assert isinstance(H, nx.Graph)
    assert len(alpha_shape.alpha_shape) == 75
    assert isinstance(alpha_shape.alpha_shape, list)
    assert alpha_shape.dimension == 2
    assert isinstance(alpha_shape.region, Region2D)
    assert alpha_shape.region_measure == 65.0
    assert alpha_shape.subregion_measure == 73.81471965135825
    assert alpha_shape.n_points_alpha_shape == 33
    assert alpha_shape.n_points_alpha_shape_rel == 1.0
    assert alpha_shape.n_points_on_boundary == 28
    assert alpha_shape.n_points_on_boundary_rel == 0.8484848484848485
    assert np.array_equal(
        alpha_shape.vertex_indices,
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
        ],
    )
    assert alpha_shape.vertices.shape == (28, 2)
    assert alpha_shape.vertices_alpha_shape.shape == (33, 2)
    assert len(alpha_shape.vertex_alpha_shape_indices) == 33
    cc = alpha_shape.connected_components
    assert len(cc) == 1
    assert isinstance(cc[0], Region2D)
    assert sorted(alpha_shape.vertices_connected_components_indices[0]) == sorted(
        [
            26,
            20,
            32,
            9,
            4,
            8,
            25,
            24,
            13,
            7,
            23,
            31,
            22,
            29,
            10,
            6,
            11,
            15,
            27,
            5,
            21,
            17,
            1,
            16,
            0,
            18,
            2,
            12,
            28,
            19,
            3,
            14,
            30,
        ]
    )
