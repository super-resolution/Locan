import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.data.hulls import BoundingBox, ConvexHull, _ConvexHullScipy, _ConvexHullShapely, \
    update_convex_hulls_in_collection
from surepy.data.cluster.clustering import cluster_dbscan


@pytest.fixture()
def locdata_simple():
    dict_ = {
        'position_x': [0, 0, 1, 4, 1, 5],
        'position_y': [0, 1, 3, 4, 3, 1]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict_))


def test_Bounding_box(locdata_simple):
    true_hull = np.array([[0, 0], [5, 4]])
    hull = BoundingBox(locdata_simple.coordinates)
    # print(locdata_simple.properties['Region_measure_bb'])
    # print(hull.region_measure)
    # print(hull.width)
    assert(hull.region_measure == 20)
    assert(hull.subregion_measure == 18)
    np.testing.assert_array_equal(hull.hull, true_hull)
    assert hull.region.region_measure == 20


def test_Convex_hull_scipy_small():
    dict_ = {
        'position_x': [0, 0, 0],
        'position_y': [0, 1, 1]
    }
    locdata = LocData(dataframe=pd.DataFrame.from_dict(dict_))
    with pytest.raises(TypeError):
        _ConvexHullScipy(locdata.coordinates)


def test_update_convex_hulls_in_collection(locdata_blobs_2d):
    clust = cluster_dbscan(locdata_blobs_2d, eps=100, min_samples=4)
    assert (len(clust) == 5)

    new_collection = update_convex_hulls_in_collection(clust, copy=False)
    assert new_collection is None
    # print(clust.references[0].properties)
    assert 'region_measure_ch' in clust.references[0].properties
    assert 'localization_density_ch' in clust.references[0].properties


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
