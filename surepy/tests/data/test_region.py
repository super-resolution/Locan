import pytest
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from surepy import LocData
from surepy.data.region import RoiRegion


# fixtures

@pytest.fixture()
def locdata():
    locdata_dict = {
        'position_x': [0, 1, 2, 3, 0, 1, 4, 5],
        'position_y': [0, 1, 2, 3, 1, 4, 5, 1],
        'position_z': [0, 1, 2, 3, 4, 4, 4, 5]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


@pytest.fixture()
def points():
    return np.array([[0, 0], [0.5, 0.5], [100, 100], [1, 1], [1.1, 0.9]])


# tests

def test_RoiRegion(points):
    region_dict = dict(
        interval=dict(region_type='interval', region_specs=(0, 1)),
        rectangle=dict(region_type='rectangle', region_specs=((0, 0), 2, 1, 10)),
        ellipse=dict(region_type='ellipse', region_specs=((1, 1), 2, 1, 10)),
        polygon=dict(region_type='polygon', region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))),
        polygon_fail=dict(region_type='polygon', region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5)))
    )

    rr = RoiRegion(**region_dict['interval'])
    assert str(rr) == str({'region_type': 'interval', 'region_specs': (0, 1)})
    assert len(rr.contains(points[:, 0])) == 1
    assert np.allclose(rr.polygon, region_dict['interval']['region_specs'])
    assert rr.dimension == 1
    assert rr.centroid == 0.5
    assert rr.max_distance == 1
    assert rr.region_measure == 1
    assert rr.subregion_measure is None

    rr = RoiRegion(**region_dict['rectangle'])
    assert str(rr) == str({'region_type': 'rectangle', 'region_specs': ((0, 0), 2, 1, 10)})
    assert len(rr.contains(points)) == 3
    assert len(rr.polygon) == 5
    assert rr.dimension == 2
    assert rr.centroid != (0, 0)
    assert rr.max_distance == np.sqrt(5)
    assert rr.region_measure == 2
    assert rr.subregion_measure == 6
    assert isinstance(rr.to_shapely(), Polygon)

    rr = RoiRegion(**region_dict['ellipse'])
    assert str(rr) == str({'region_type': 'ellipse', 'region_specs': ((1, 1), 2, 1, 10)})
    assert len(rr.contains(points)) == 3
    assert len(rr.polygon) == 26
    assert rr.dimension == 2
    assert rr.centroid == (1, 1)
    assert rr.max_distance == 2
    assert rr.region_measure == 1.5707963267948966
    assert rr.subregion_measure == 4.844224108065043
    assert isinstance(rr.to_shapely(), Polygon)

    rr = RoiRegion(**region_dict['polygon'])
    assert str(rr) == str({'region_type': 'polygon', 'region_specs': ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))})
    assert len(rr.contains(points)) == 1
    assert len(rr.polygon) == 5
    assert rr.dimension == 2
    assert rr.centroid
    assert rr.max_distance == np.sqrt(2)
    assert rr.region_measure == 0.75
    assert rr.subregion_measure == 3.618033988749895
    assert isinstance(rr.to_shapely(), Polygon)


def test_RoiRegion_RoiPolygon_error():
    with pytest.raises(ValueError):
        RoiRegion(region_type='polygon', region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5)))


# This test is just for visualization
# def test_show_regions_as_artist(points):
#     roiI = Roi(reference=locdata, region_specs=(0, 0), region_type='interval')
#     roiR = Roi(reference=locdata, region_specs=((0, 0), 2, 1, 0), region_type='rectangle')
#     roiE = Roi(reference=locdata, region_specs=((1, 1), 2, 1, 0), region_type='ellipse')
#     assert len(roiE._region.contains(points)) == 2
#     roiP = Roi(reference=locdata, region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0)),
#                region_type='polygon')
#     fig, ax = plt.subplots()
#     ax.scatter(points[:,0], points[:,1])
#     # ax.add_patch(roiI._region.as_artist())
#     ax.add_patch(roiR._region.as_artist(fill=False))
#     ax.add_patch(roiE._region.as_artist(fill=False))
#     ax.add_patch(roiP._region.as_artist(fill=False))
#     ax.set_xlim([0, 3])
#     ax.set_ylim([0, 3])
#     plt.show()
