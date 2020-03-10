import pytest
import numpy as np
import matplotlib.pyplot as plt  # needed for visualization
from scipy.spatial import Delaunay
import networkx as nx

from surepy import AlphaShape, AlphaComplex
from surepy.data.hulls.alpha_shape import _circumcircle, _half_distance


def test__circumcircle_2d(locdata_2d):
    points = np.array([(0, 0), (1, 1+np.sqrt(2)), (1+np.sqrt(2), 1)])
    center, radius = _circumcircle(points, [2, 1, 0])
    assert radius == np.sqrt(2)
    assert np.array_equal(center, [1, 1])

    points = locdata_2d.coordinates
    triangulation = Delaunay(points)
    center, radius = _circumcircle(points, triangulation.simplices[0])
    assert radius == 1.8210786221487993
    assert center[0] == 3.357142857142857


def test__half_distance():
    points = np.array([(0, 0), (0, 1)])
    radius = _half_distance(points)
    assert radius == 0.5


def test__AlphaComplex_1():
    points = np.array([[10, 10], [10, 14], [11, 11], [12, 11], [13, 10], [13, 13], [15, 14], [17, 16],
                       [10, 24]])
    assert len(points) == 9
    alpha_complex = AlphaComplex(points)
    assert len(alpha_complex.alpha_complex) == 19
    assert alpha_complex.dimension == 2

    alpha = 2.2
    ac_simplices_all = alpha_complex.get_alpha_complex_all(alpha)
    ac_simplices_exterior = alpha_complex.get_alpha_complex_exterior(alpha)
    ac_simplices_interior = alpha_complex.get_alpha_complex_interior(alpha)
    ac_simplices_regular = alpha_complex.get_alpha_complex_regular(alpha)
    ac_simplices_singular = alpha_complex.get_alpha_complex_singular(alpha)

    lengths = [len(sim) for sim in [
        ac_simplices_all, ac_simplices_exterior, ac_simplices_interior, ac_simplices_regular, ac_simplices_singular]]
    assert lengths == [13, 6, 5, 5, 3]

    H = alpha_complex.to_graph(alpha, type='regular')
    assert isinstance(H, nx.Graph)
    # visualization
    nx.draw_networkx(H, points, with_labels=True)
    # plt.show()


def test__AlphaComplex_2():
    points = np.array([[10, 10], [10, 14], [11, 11], [12, 11], [13, 10], [13, 13], [15, 14], [17, 16],
                       [20, 20], [20, 24], [21, 21], [22, 21], [23, 20], [23, 23], [25, 24],
                       [10, 24]])
    assert len(points) == 16
    alpha_complex = AlphaComplex(points)
    assert len(alpha_complex.alpha_complex) == 38
    assert alpha_complex.dimension == 2

    alpha = 2.2
    ac_simplices_all = alpha_complex.get_alpha_complex_all(alpha)
    ac_simplices_exterior = alpha_complex.get_alpha_complex_exterior(alpha)
    ac_simplices_interior = alpha_complex.get_alpha_complex_interior(alpha)
    ac_simplices_regular = alpha_complex.get_alpha_complex_regular(alpha)
    ac_simplices_singular = alpha_complex.get_alpha_complex_singular(alpha)

    lengths = [len(sim) for sim in [
        ac_simplices_all, ac_simplices_exterior, ac_simplices_interior, ac_simplices_regular, ac_simplices_singular]]
    assert lengths == [25, 13, 10, 10, 5]

    H = alpha_complex.to_graph(alpha, type='regular')
    assert isinstance(H, nx.Graph)
    # visualization
    nx.draw_networkx(H, points, with_labels=True)
    # plt.show()


def test__AlphaComplex_3():
    points = np.array([[10, 10], [10, 20], [20, 20], [20, 10], [10, 15], [15, 20], [20, 15], [15, 10],
                       [10, 12], [10, 18], [20, 18], [20, 12], [17, 20], [12, 10], [17, 10], [12, 20],

                       [11, 11], [11, 19], [19, 19], [19, 11], [11, 15], [15, 19], [19, 15], [15, 11],
                       [13, 11], [12, 12], [12, 18], [13, 19], [17, 19], [18, 18], [17, 11], [15, 11], [18, 12],
                       [15, 15]
                       ])
    assert len(points) == 34
    alpha_complex = AlphaComplex(points)
    assert len(alpha_complex.alpha_complex) == 80
    assert alpha_complex.dimension == 2

    alpha = 2.2
    ac_simplices_all = alpha_complex.get_alpha_complex_all(alpha)
    ac_simplices_exterior = alpha_complex.get_alpha_complex_exterior(alpha)
    ac_simplices_interior = alpha_complex.get_alpha_complex_interior(alpha)
    ac_simplices_regular = alpha_complex.get_alpha_complex_regular(alpha)
    ac_simplices_singular = alpha_complex.get_alpha_complex_singular(alpha)

    lengths = [len(sim) for sim in [
        ac_simplices_all, ac_simplices_exterior, ac_simplices_interior, ac_simplices_regular, ac_simplices_singular]]
    assert lengths == [77, 3, 40, 28, 9]

    H = alpha_complex.to_graph(alpha=2.2, type='regular')
    assert isinstance(H, nx.Graph)
    # visualization
    nx.draw_networkx(H, points, with_labels=True)
    # plt.show()


def test__AlphaShape_1():
    points = np.array([[10, 10], [10, 14], [11, 11], [12, 11], [13, 10], [13, 13], [15, 14], [17, 16],
                       [10, 24]])
    assert len(points) == 9

    alpha = 2.2
    alpha_shape = AlphaShape(points, alpha)
    H = alpha_shape.alpha_complex.to_graph(alpha, type='regular')
    assert isinstance(H, nx.Graph)
    # visualization
    nx.draw_networkx(H, points, with_labels=True)
    # plt.show()

    assert len(alpha_shape.alpha_shape) == 13
    assert isinstance(alpha_shape.alpha_shape, list)
    assert alpha_shape.dimension == 2
    assert alpha_shape.hull
    assert alpha_shape.region_measure == 8.5
    assert alpha_shape.subregion_measure == 13.738768882709854
    assert alpha_shape.n_points_alpha_shape == 8
    assert alpha_shape.n_points_alpha_shape_rel == 0.8888888888888888
    assert alpha_shape.n_points_on_boundary == 5
    assert alpha_shape.n_points_on_boundary_rel == 0.625
    assert np.array_equal(alpha_shape.vertex_indices, [0, 1, 2, 4, 5])
    assert alpha_shape.vertices.shape == (5, 2)


def test_AlphaShape_2():
    points = np.array([[10, 10], [10, 14], [11, 11], [12, 11], [13, 10], [13, 13], [15, 14], [17, 16],
                       [20, 20], [20, 24], [21, 21], [22, 21], [23, 20], [23, 23], [25, 24],
                       [10, 24]])
    assert len(points) == 16

    alpha = 2.2
    alpha_shape = AlphaShape(points, alpha)
    H = alpha_shape.alpha_complex.to_graph(alpha, type='regular')
    assert isinstance(H, nx.Graph)
    # visualization
    nx.draw_networkx(H, points, with_labels=True)
    # plt.show()

    assert len(alpha_shape.alpha_shape) == 25
    assert isinstance(alpha_shape.alpha_shape, list)
    assert alpha_shape.dimension == 2
    assert alpha_shape.hull
    assert alpha_shape.region_measure == 17
    assert alpha_shape.subregion_measure == 27.47753776541971
    assert alpha_shape.n_points_alpha_shape == 15
    assert alpha_shape.n_points_alpha_shape_rel == 0.9375
    assert alpha_shape.n_points_on_boundary == 10
    assert alpha_shape.n_points_on_boundary_rel == 0.6666666666666666
    assert np.array_equal(alpha_shape.vertex_indices, [0, 1, 2, 4, 5, 8, 9, 10, 12, 13])
    assert alpha_shape.vertices.shape == (10, 2)


def test_AlphaShape_3():
    points = np.array([[10, 10], [10, 20], [20, 20], [20, 10], [10, 15], [15, 20], [20, 15], [15, 10],
                       [10, 12], [10, 18], [20, 18], [20, 12], [17, 20], [12, 10], [17, 10], [12, 20],

                       [11, 11], [11, 19], [19, 19], [19, 11], [11, 15], [15, 19], [19, 15], [15, 11],
                       [13, 11], [12, 12], [12, 18], [13, 19], [17, 19], [18, 18], [17, 11], [15, 11], [18, 12],
                       [15, 15]
                       ])
    assert len(points) == 34

    alpha = 2.2
    alpha_shape = AlphaShape(points, alpha)
    H = alpha_shape.alpha_complex.to_graph(alpha, type='regular')
    assert isinstance(H, nx.Graph)
    # visualization
    nx.draw_networkx(H, points, with_labels=True)
    # plt.show()

    assert len(alpha_shape.alpha_shape) == 77
    assert isinstance(alpha_shape.alpha_shape, list)
    assert alpha_shape.dimension == 2
    assert alpha_shape.hull
    assert alpha_shape.region_measure == 48
    assert alpha_shape.subregion_measure == 66.3059648901659
    assert alpha_shape.n_points_alpha_shape == 33
    assert alpha_shape.n_points_alpha_shape_rel == 0.9705882352941176
    assert alpha_shape.n_points_on_boundary == 28
    assert alpha_shape.n_points_on_boundary_rel == 0.8484848484848485
    assert np.array_equal(alpha_shape.vertex_indices, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20,
                                                       21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32])
    assert alpha_shape.vertices.shape == (28, 2)


@pytest.mark.skip('Requires visual inspection.')
def test__AlphaComplex_visual(locdata_2d):
    points = locdata_2d.coordinates
    alpha_complex = AlphaComplex(points)

    alpha = 1.6
    ac_simplices_all = alpha_complex.get_alpha_complex_all(alpha)
    ac_simplices_exterior = alpha_complex.get_alpha_complex_exterior(alpha)
    ac_simplices_interior = alpha_complex.get_alpha_complex_interior(alpha)
    ac_simplices_regular = alpha_complex.get_alpha_complex_regular(alpha)
    ac_simplices_singular = alpha_complex.get_alpha_complex_singular(alpha)

    lengths = [len(sim) for sim in [
        ac_simplices_all, ac_simplices_exterior, ac_simplices_interior, ac_simplices_regular, ac_simplices_singular]]
    assert lengths == [8, 2, 0, 6, 2]

    locdata_2d.data.plot(*locdata_2d.coordinate_labels, kind='scatter')
    for simp in ac_simplices_all:
        plt.plot(points[simp, 0], points[simp, 1], '-b')
    for simp in ac_simplices_interior:
        plt.plot(points[simp, 0], points[simp, 1], '--g')
    for simp in ac_simplices_regular:
        plt.plot(points[simp, 0], points[simp, 1], '--r')
    for simp in ac_simplices_singular:
        plt.plot(points[simp, 0], points[simp, 1], '--y')
    plt.show()
