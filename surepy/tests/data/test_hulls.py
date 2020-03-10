import pytest
import numpy as np
from surepy import BoundingBox, ConvexHull
from surepy.data.hulls.hull import _ConvexHullScipy, _ConvexHullShapely


def test_BoundingBox_2d(locdata_2d):
    true_hull = np.array([[1, 1], [5, 6]])
    hull = BoundingBox(locdata_2d.coordinates)
    assert hull.width.shape == (2,)
    assert(locdata_2d.properties['region_measure_bb'] == hull.region_measure == 20)
    assert(hull.subregion_measure == 18)
    np.testing.assert_array_equal(hull.hull, true_hull)
    assert hull.region.region_measure == 20


def test_BoundingBox_3d(locdata_3d):
    true_hull = np.array([[1, 1, 1], [5, 6, 5]])
    hull = BoundingBox(locdata_3d.coordinates)
    assert hull.width.shape == (3,)
    assert(hull.region_measure == 80)
    assert(hull.subregion_measure == 26)
    np.testing.assert_array_equal(hull.hull, true_hull)
    with pytest.raises(NotImplementedError):
        assert hull.region.region_measure == 80


@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', 0),
    ('locdata_single_localization', 0),
    ('locdata_non_standard_index', 20)
])
def test_BoundingBox(locdata_empty, locdata_single_localization, locdata_non_standard_index,
                     fixture_name, expected):
    locdata = eval(fixture_name)
    hull = BoundingBox(locdata.coordinates)
    assert hull.region_measure == expected


def test_ConvexHullScipy(locdata_2d):
    true_convex_hull_indices = np.array([5, 3, 1, 0, 4])
    hull = _ConvexHullScipy(locdata_2d.coordinates)
    assert np.array_equal(hull.vertex_indices, true_convex_hull_indices)
    assert np.array_equal(hull.vertices, locdata_2d.coordinates[true_convex_hull_indices])
    assert hull.points_on_boundary == 5
    assert hull.region_measure == 14
    assert hull.region.region_measure == 14


def test_ConvexHullShapely(locdata_2d):
    true_convex_hull_indices = np.array([0, 1, 3, 5, 4])
    hull = _ConvexHullShapely(locdata_2d.coordinates)
    # assert np.array_equal(hull.vertex_indices, true_convex_hull_indices)
    assert np.array_equal(hull.vertices, locdata_2d.coordinates[true_convex_hull_indices])
    assert hull.points_on_boundary == 5
    assert hull.region_measure == 14
    assert hull.region.region_measure == 14


def test_ConvexHull_2d(locdata_2d):
    true_convex_hull_indices = np.array([5, 3, 1, 0, 4])
    hull = ConvexHull(locdata_2d.coordinates, method='scipy')
    assert np.array_equal(hull.vertex_indices, true_convex_hull_indices)
    assert np.array_equal(hull.vertices, locdata_2d.coordinates[true_convex_hull_indices])
    assert hull.points_on_boundary == 5
    assert hull.region_measure == 14
    assert hull.region.region_measure == 14

    true_convex_hull_indices = np.array([0, 1, 3, 5, 4])
    hull = ConvexHull(locdata_2d.coordinates, method='shapely')
    # assert np.array_equal(hull.vertex_indices, true_convex_hull_indices)
    assert np.array_equal(hull.vertices, locdata_2d.coordinates[true_convex_hull_indices])
    assert hull.points_on_boundary == 5
    assert hull.region_measure == 14
    assert hull.region.region_measure == 14


def test_ConvexHull_3d(locdata_3d):
    true_convex_hull_indices = np.array([0, 1, 2, 3, 4, 5])
    hull = ConvexHull(locdata_3d.coordinates, method='scipy')
    assert np.array_equal(hull.vertex_indices, true_convex_hull_indices)
    assert np.array_equal(hull.vertices, locdata_3d.coordinates[true_convex_hull_indices])
    assert hull.points_on_boundary == 6
    assert hull.region_measure == 19.833333333333332
    with pytest.raises(NotImplementedError):
        assert hull.region.region_measure == 19.833333333333332

    with pytest.raises(TypeError):
        ConvexHull(locdata_3d.coordinates, method='shapely')


@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', 0),
    ('locdata_single_localization', 0)
])
def test_ConvexHull(locdata_empty, locdata_single_localization,
                    fixture_name, expected):
    locdata = eval(fixture_name)
    with pytest.raises(TypeError):
        hull = ConvexHull(locdata.coordinates, method='scipy')
    with pytest.raises(TypeError):
        hull = ConvexHull(locdata.coordinates, method='shapely')
