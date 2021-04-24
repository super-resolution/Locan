import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # needed for visual inspection
from shapely.geometry import Polygon, MultiPolygon

from locan import LocData, RoiRegion
from locan.data.region import _ShapelyPolygon, _ShapelyMultiPolygon


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
    assert np.allclose(rr.polygon.astype(float), region_dict['interval']['region_specs'])
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


def test_RoiRegion_shapely(points):
    rr = RoiRegion(region_type='shapelyPolygon', region_specs=[((0, 0), (0, 1), (1, 1), (1, 0.5)), []])
    assert str(rr) == str({'region_type': 'shapelyPolygon', 'region_specs': [((0, 0), (0, 1), (1, 1), (1, 0.5)), []]})
    assert len(rr.contains(points)) == 1
    assert len(rr.polygon) == 2
    assert rr.dimension == 2
    assert rr.centroid
    assert rr.max_distance == np.sqrt(2)
    assert rr.region_measure == 0.75
    assert rr.subregion_measure == 3.618033988749895
    assert isinstance(rr.to_shapely(), Polygon)

    rr = RoiRegion(region_type='shapelyMultiPolygon', region_specs=[
        [[(0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0)], []],
        [[(2, 0), (2, 1), (3, 1), (3, 0.5), (2, 0)], []]
    ])
    assert len(rr.contains(points)) == 1
    assert len(rr.polygon) == 2
    assert rr.dimension == 2
    assert rr.centroid
    assert rr.max_distance == 3.1622776601683795
    assert rr.region_measure == 1.5
    assert rr.subregion_measure == 7.23606797749979
    assert isinstance(rr.to_shapely(), MultiPolygon)


def test__ShapelyPolygon():
    # Polygon with holes
    region_specs = [[(0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0)],
                    [[(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0), (0, 0)]]]
    pol = Polygon(*region_specs)
    shapely_polygon = _ShapelyPolygon.from_shapely(pol)
    assert shapely_polygon.region_specs[1][0][1][1] == 0.5
    # print(shapely_polygon.polygon)
    # [list([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.5], [0.0, 0.0]])
    # print(type(shapely_polygon.polygon))
    #  list([[[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.5, 0.0], [0.0, 0.0]]])]

    shapely_polygon = RoiRegion.from_shapely(region_type='shapelyPolygon', shapely_obj=pol)
    assert shapely_polygon.region_specs[1][0][1][1] == 0.5


def test__ShapelyMultiPolygon():
    # MultiPolygon with holes
    region_specs = [
        [[(0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0)], [[(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0), (0, 0)]]],
        [[(2, 0), (2, 1), (3, 1), (3, 0.5), (2, 0)], []]
    ]
    mp = MultiPolygon(region_specs)
    shapely_mpolygon = _ShapelyMultiPolygon.from_shapely(mp)
    assert shapely_mpolygon.region_specs[1][0][0][0] == 2

    shapely_mpolygon = RoiRegion.from_shapely(region_type='shapelyMultiPolygon', shapely_obj=mp)
    assert shapely_mpolygon.region_specs[1][0][0][0] == 2


def test_RoiRegion_RoiPolygon_error():
    with pytest.raises(ValueError):
        RoiRegion(region_type='polygon', region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5)))


def test_show_regions_as_artist_(points):
    roiI = RoiRegion(region_specs=(0, 0), region_type='interval')
    roiR = RoiRegion(region_specs=((0, 0), 2, 1, 0), region_type='rectangle')
    roiE = RoiRegion(region_specs=((1, 1), 2, 1, 0), region_type='ellipse')
    assert len(roiE._region.contains(points)) == 2
    roiP = RoiRegion(region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0)), region_type='polygon')
    roi_sP = RoiRegion(region_specs=[((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))], region_type='shapelyPolygon')
    roi_sMP = RoiRegion(region_specs=[[((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0)), []], []],
                        region_type='shapelyMultiPolygon')
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1])
    # ax.add_patch(roiI._region.as_artist())   # not implemented
    ax.add_patch(roiR.as_artist(origin=(-1, 0), fill=False))
    ax.add_patch(roiE.as_artist(origin=(-1, 0), fill=False))
    ax.add_patch(roiP.as_artist(origin=(-1, 0), fill=False))
    ax.add_patch(roi_sP.as_artist(origin=(-0.5, 0), fill=False))
    for pat in roi_sMP.as_artist(origin=(-0.1, 0), fill=False):
        ax.add_patch(pat)
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 4])
    # plt.show()
