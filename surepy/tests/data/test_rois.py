import warnings
import pytest
import numpy as np
import pandas as pd

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.io.io_locdata import load_txt_file
from surepy.data.rois import RoiRegion, Roi, load_from_roi_file


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


def test_Roi(locdata):
    roi = Roi(region_specs=((1, 1), 10, 10, 0.), type='rectangle')
    assert(repr(roi)=='Roi(reference=None, region_specs=((1, 1), 10, 10, 0.0), type=rectangle, meta=)')
    assert(roi.reference is None)
    assert(roi.region_specs == ((1, 1), 10, 10, 0.))
    assert(roi.meta.file_path == '')
    assert(roi.meta.file_type == 0)
    del(roi)

    roi = Roi(reference=locdata, region_specs=((1, 1), 10, 10, 0.0), type='rectangle')
    assert(roi.reference is locdata)
    assert(roi.region_specs == ((1, 1), 10, 10, 0.))
    assert(roi.meta.file_path == '')
    assert(roi.meta.file_type == 0)
    # print(locdata.meta)
    # print(roi)
    # print(True if locdata.meta.file_path else False)
    # print(locdata.meta.file_type)
    # print()
    del roi

    roi = Roi(reference=locdata, region_specs=((1, 1), 10, 10, 0.), type='rectangle',
              meta=dict(file_path='my/path/to/file', file_type=0))
    assert(roi.meta.file_path == 'my/path/to/file')
    assert(roi.meta.file_type == 0)
    del roi

    roi = Roi(reference=True, region_specs=((1, 1), 10, 10, 0.), type='rectangle',
              meta=dict(file_path='my/path/to/file', file_type=0))
    assert(roi.meta.file_path == 'my/path/to/file')
    assert(roi.meta.file_type == 0)
    del roi


def test_Roi_locdata(locdata):
    roi = Roi(reference=locdata, region_specs=((0, 0), 3, 3, 0.), type='rectangle')
    dat_1 = roi.locdata()
    assert(len(dat_1) == 5)
    del roi

    roi = Roi(reference=True, region_specs=((1, 1), 500, 500, 0.), type='rectangle',
              meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1))
    dat_1 = roi.locdata()
    assert(len(dat_1) == 5)
    del roi

    roi = Roi(reference=None, region_specs=((1, 1), 500, 500, 0.), type='rectangle',
              meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1))
    dat_1 = roi.locdata()
    assert(dat_1 is None)
    del roi


def test_Roi_io(locdata):
    path = ROOT_DIR + '/tests/test_data/roi.yaml'

    roi = Roi(reference=locdata, region_specs=((1, 1), 500, 500, 0.), type='rectangle')
    with pytest.warns(UserWarning):
        roi.to_yaml(path=path)

    roi_new = Roi().from_yaml(path = path)
    assert(roi_new.reference == None)

    dat_1 = roi_new.locdata()
    assert(dat_1==None)
    del(roi, dat_1)

    roi = Roi(reference=True, region_specs=((1, 1), 500, 500, 0.), type='rectangle',
              meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1))
    roi.to_yaml(path=path)
    roi_new = Roi().from_yaml(path = path)
    assert(roi_new.reference == True)
    dat_1 = roi_new.locdata()
    assert(len(dat_1)==5)
    del(roi, dat_1)

    roi = Roi(reference=locdata, region_specs=((1, 1), 500, 500, 0.), type='rectangle',
              meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1))
    roi.to_yaml(path=path)
    roi_new = Roi().from_yaml(path = path)
    print(roi_new)
    assert(roi_new.reference == True)
    dat_1 = roi_new.locdata()
    assert(len(dat_1)==5)
    del(roi, dat_1)


def test_roi_locdata_from_file():
    dat = load_txt_file(path=ROOT_DIR + '/tests/test_data/five_blobs.txt')

    roi_1 = Roi(reference=dat, region_specs=((1, 1), 500, 500, 0.), type='rectangle')
    dat_1 = roi_1.locdata()
    assert(len(dat_1) == 5)

    roi_2 = Roi(reference=True, meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1),
                region_specs=((1, 1), 500, 500, 0.),
                type='rectangle')
    dat_2 = roi_2.locdata()
    assert(len(dat_2) == 5)


def test_load_from_roi_file():
    path = ROOT_DIR + '/tests/test_data/roi.yaml'

    roi = Roi(reference=True, region_specs=((1, 1), 500, 500, 0.), type='rectangle',
              meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1))
    roi.to_yaml(path=path)

    dat = load_from_roi_file(path)
    assert(len(dat) == 5)

    dat_2 = load_from_roi_file(path, meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1))
    assert(len(dat_2) == 5)


