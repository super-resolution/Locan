import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.io.io_locdata import load_txt_file
from surepy.data.region import RoiRegion
from surepy.data.rois import Roi, select_by_drawing
from surepy.data import metadata_pb2


# fixtures

@pytest.fixture()
def locdata():
    dict = {
        'Position_x': [0, 1, 2, 3, 0, 1, 4, 5],
        'Position_y': [0, 1, 2, 3, 1, 4, 5, 1],
        'Position_z': [0, 1, 2, 3, 4, 4, 4, 5]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))

@pytest.fixture()
def points():
    return np.array([[0, 0], [0.5, 0.5], [100, 100], [1, 1], [1.1, 0.9]])

# tests

def test_RoiRegion(points):
    region_dict = dict(
        interval=dict(region_type='interval', region_specs=(0,1)),
        rectangle=dict(region_type='rectangle', region_specs=((0, 0), 2, 1, 10)),
        ellipse=dict(region_type='ellipse', region_specs=((1, 1), 2, 1, 10)),
        polygon=dict(region_type='polygon', region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))),
        polygon_fail=dict(region_type='polygon', region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5)))
    )

    rr = RoiRegion(**region_dict['interval'])
    assert str(rr) == str({'region_type': 'interval', 'region_specs': (0,1)})
    assert len(rr.contains(points[:,0])) == 1
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

    rr = RoiRegion(**region_dict['ellipse'])
    assert str(rr) == str({'region_type': 'ellipse', 'region_specs': ((1, 1), 2, 1, 10)})
    assert len(rr.contains(points)) == 3
    assert len(rr.polygon) == 26
    assert rr.dimension == 2
    assert rr.centroid == (1, 1)
    assert rr.max_distance == 2
    assert rr.region_measure == 1.5707963267948966
    assert rr.subregion_measure == 4.844224108065043

    rr = RoiRegion(**region_dict['polygon'])
    assert str(rr) == str({'region_type': 'polygon', 'region_specs': ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))})
    assert len(rr.contains(points)) == 1
    assert len(rr.polygon) == 5
    assert rr.dimension == 2
    assert rr.centroid
    assert rr.max_distance == np.sqrt(2)
    assert rr.region_measure == 0.75
    assert rr.subregion_measure == 3.618033988749895


def test_RoiRegion_RoiPolygon_error():
    with pytest.raises(ValueError):
        rr = RoiRegion(region_type='polygon', region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5)))


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


def test_Roi(locdata):
    roi = Roi(reference=locdata, region_specs=((0, 0), 2, 1, 10), region_type='rectangle')
    new_dat = roi.locdata()
    assert len(new_dat) == 2


def test_Roi_io(locdata):
    path = ROOT_DIR + '/tests/test_data/roi.yaml'

    roi = Roi(region_specs=((0, 0), 2, 1, 10), region_type='rectangle')
    roi.to_yaml(path=path)

    roi = Roi(reference=locdata, region_type='rectangle', region_specs=((0, 0), 2, 1, 10))
    with pytest.warns(UserWarning):
        roi.to_yaml(path=path)

    roi_new = Roi.from_yaml(path=path)
    assert roi_new.reference is None

    roi = Roi(reference=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1),
              region_type='rectangle', region_specs=((0, 0), 2, 1, 10))
    assert isinstance(roi.reference, metadata_pb2.Metadata)
    roi.to_yaml(path=path)

    roi_new = Roi.from_yaml(path=path)
    assert roi_new

    locdata_2 = LocData.from_selection(locdata,
                                       meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1))
    roi = Roi(reference=locdata_2,
              region_type='rectangle', region_specs=((0, 0), 2, 1, 10))
    assert isinstance(roi.reference.meta, metadata_pb2.Metadata)
    roi.to_yaml(path=path)

    roi_new = Roi.from_yaml(path=path)
    assert roi_new
    assert isinstance(roi_new.reference, metadata_pb2.Metadata)


def test_roi_locdata_from_file(locdata):
    dat = load_txt_file(path=ROOT_DIR + '/tests/test_data/five_blobs.txt')

    roi = Roi(reference=locdata, region_type='rectangle', region_specs=((0, 0), 2.5, 1, 10))
    dat_1 = roi.locdata()
    assert(len(dat_1) == 2)

    roi = Roi(reference=locdata, region_type='rectangle', region_specs=((0, 0), 2.5, 1, 10),
              properties_for_roi=['Position_y', 'Position_z'])
    dat_1 = roi.locdata()
    assert(len(dat_1) == 1)


def test_as_artist():
    roiR = Roi(reference=locdata, region_specs=((0, 0), 0.5, 0.5, 10), region_type='rectangle')
    roiE = Roi(reference=locdata, region_specs=((0.5, 0.5), 0.2, 0.3, 10), region_type='ellipse')
    roiP = Roi(reference=locdata, region_specs=((0, 0.8), (0.1, 0.9), (0.6, 0.9), (0.7, 0.7), (0, 0.8)),
               region_type='polygon')
    fig, ax = plt.subplots()
    ax.add_patch(roiR._region.as_artist())
    ax.add_patch(roiE._region.as_artist())
    ax.add_patch(roiP._region.as_artist())
    # plt.show()


def test_select_by_drawing():
    dat = load_txt_file(path=ROOT_DIR + '/tests/test_data/five_blobs.txt')
    # select_by_drawing(dat, type='rectangle')
    # select_by_drawing(dat, type='ellipse')
    # todo: fix bug in polygon selector
    # select_by_drawing(dat, type='polygon')
