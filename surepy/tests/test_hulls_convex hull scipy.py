import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.data.hulls import  Bounding_box, Convex_hull_scipy


@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


def test_Bounding_box(locdata_simple):
    true_hull = np.array([[0, 0], [5, 4]])
    H = Bounding_box(locdata_simple.coordinates)
    print(locdata_simple.properties['Region_measure_bb'])
    print(H.region_measure)
    # print(H.width)
    assert(H.region_measure == 20)
    assert(H.subregion_measure == 18)
    np.testing.assert_array_equal(H.hull, true_hull)


def test_Convex_hull_scipy(locdata_simple):
    true_convex_hull_indices = np.array([0, 4, 3, 2, 1])
    H = Convex_hull_scipy(locdata_simple.coordinates)
    np.testing.assert_array_equal(H.vertex_indices, true_convex_hull_indices)
    assert (H.region_measure == 14.659642811429332)

# def test_Convex_hull_scipy(few_random_points):
#     true_convex_hull_indices = np.array([1, 4, 0, 5, 9])
#     H = Convex_hull_scipy(few_random_points)
#     np.testing.assert_array_equal(H.vertex_indices, true_convex_hull_indices)
#     assert (H.region_measure == 2.6960242565501966)
