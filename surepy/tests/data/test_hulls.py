import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.data.hulls import  BoundingBox, ConvexHull, ConvexHullShapely, update_convex_hulls_in_collection
from surepy.data.cluster.clustering import cluster_dbscan


@pytest.fixture()
def locdata_simple():
    dict = {
        'position_x': [0, 0, 1, 4, 1, 5],
        'position_y': [0, 1, 3, 4, 3, 1]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


def test_Bounding_box(locdata_simple):
    true_hull = np.array([[0, 0], [5, 4]])
    H = BoundingBox(locdata_simple.coordinates)
    # print(locdata_simple.properties['Region_measure_bb'])
    # print(H.region_measure)
    # print(H.width)
    assert(H.region_measure == 20)
    assert(H.subregion_measure == 18)
    np.testing.assert_array_equal(H.hull, true_hull)
    assert H.region.region_measure == 20


def test_Convex_hull_scipy(locdata_simple):
    true_convex_hull_indices = np.array([0, 5, 3, 2, 1])
    H = ConvexHull(locdata_simple.coordinates)
    np.testing.assert_array_equal(H.vertex_indices, true_convex_hull_indices)
    assert H.region_measure == 12.5
    assert H.region.region_measure == 12.5


def test_Convex_hull_shapely(locdata_simple):
    true_convex_hull_indices = np.array([0, 5, 3, 2, 1])
    H = ConvexHullShapely(locdata_simple.coordinates)
    # np.testing.assert_array_equal(H.vertex_indices, true_convex_hull_indices)
    assert H.region_measure == 12.5
    # assert H.region.region_measure == 12.5


def test_Convex_hull_scipy_small():
    dict = {
        'position_x': [0, 0, 0],
        'position_y': [0, 1, 1]
    }
    locdata = LocData(dataframe=pd.DataFrame.from_dict(dict))
    with pytest.raises(TypeError):
        ConvexHull(locdata.coordinates)


def test_update_convex_hulls_in_collection(locdata_blobs_2d):
    clust = cluster_dbscan(locdata_blobs_2d, eps=100, min_samples=4)
    assert (len(clust) == 5)

    new_collection = update_convex_hulls_in_collection(clust, copy=False)
    assert new_collection is None
    # print(clust.references[0].properties)
    assert 'region_measure_ch' in clust.references[0].properties
    assert 'localization_density_ch' in clust.references[0].properties
