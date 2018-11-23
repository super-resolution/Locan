import warnings
import pytest
import pandas as pd

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.io.io_locdata import load_txt_file
from surepy.data.rois import Roi, load_from_roi_file


# fixtures

@pytest.fixture()
def locdata():
    dict = {
        'Position_x': [0, 1, 2, 3, 0, 1, 4, 5],
        'Position_y': [0, 1, 2, 3, 1, 4, 5, 1],
        'Position_z': [0, 1, 2, 3, 4, 4, 4, 5]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


# tests

def test_Roi(locdata):
    roi = Roi(points=(1, 10, 1, 10), type='rectangle')
    assert(repr(roi)=='Roi(reference=None, points=(1, 10, 1, 10), type=rectangle, meta=)')
    assert(roi.reference is None)
    assert(roi.points==(1, 10, 1, 10))
    assert(roi.meta.file_path=='')
    assert(roi.meta.file_type==0)
    del(roi)

    roi = Roi(reference=locdata, points=(1, 10, 1, 10), type='rectangle')
    assert(roi.reference is locdata)
    assert(roi.points==(1, 10, 1, 10))
    assert(roi.meta.file_path=='')
    assert(roi.meta.file_type==0)
    # print(locdata.meta)
    # print(roi)
    # print(True if locdata.meta.file_path else False)
    # print(locdata.meta.file_type)
    # print()
    del(roi)

    roi = Roi(reference=locdata, points=(1, 10, 1, 10), type='rectangle', meta=dict(file_path='my/path/to/file', file_type=0))
    assert(roi.meta.file_path=='my/path/to/file')
    assert(roi.meta.file_type==0)
    del (roi)

    roi = Roi(reference=True, points=(1, 10, 1, 10), type='rectangle', meta=dict(file_path='my/path/to/file', file_type=0))
    assert(roi.meta.file_path=='my/path/to/file')
    assert(roi.meta.file_type==0)
    del (roi)


def test_Roi_locdata(locdata):

    roi = Roi(reference=locdata, points=(0,3), type='rectangle')
    dat_1 = roi.locdata()
    assert(len(dat_1)==6)
    del (roi)

    roi = Roi(reference=locdata, points=(0,3,0,3), type='rectangle')
    dat_1 = roi.locdata()
    assert(len(dat_1)==5)
    del (roi)

    roi = Roi(reference=locdata, points=(0,3,0,3,0,3), type='rectangle')
    dat_1 = roi.locdata()
    assert(len(dat_1)==4)
    del (roi)

    roi = Roi(reference=True, points=(1, 500, 1, 500), type='rectangle',
              meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1))
    dat_1 = roi.locdata()
    assert(len(dat_1)==5)
    del (roi)

    roi = Roi(reference=None, points=(1, 500, 1, 500), type='rectangle',
              meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1))
    dat_1 = roi.locdata()
    assert(dat_1 is None)
    del (roi)


def test_Roi_io(locdata):
    path = ROOT_DIR + '/tests/test_data/roi.yaml'

    roi = Roi(reference=locdata, points=(1, 500, 1, 500), type='rectangle')
    with pytest.warns(UserWarning):
        roi.to_yaml(path=path)

    roi_new = Roi().from_yaml(path = path)
    assert(roi_new.reference == None)

    dat_1 = roi_new.locdata()
    assert(dat_1==None)
    del(roi, dat_1)

    roi = Roi(reference=True, points=(1, 500, 1, 500), type='rectangle',
              meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1))
    roi.to_yaml(path=path)
    roi_new = Roi().from_yaml(path = path)
    assert(roi_new.reference == True)
    dat_1 = roi_new.locdata()
    assert(len(dat_1)==5)
    del(roi, dat_1)

    roi = Roi(reference=locdata, points=(1, 500, 1, 500), type='rectangle',
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

    roi_1 = Roi(reference=dat, points=(1, 500, 1, 500), type='rectangle' )
    dat_1 = roi_1.locdata()
    assert(len(dat_1)==5)

    roi_2 = Roi(reference=True, meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1),
              points=(1, 500, 1, 500),
              type='rectangle')
    dat_2 = roi_2.locdata()
    assert(len(dat_2)==5)


def test_load_from_roi_file():
    path = ROOT_DIR + '/tests/test_data/roi.yaml'

    roi = Roi(reference=True, points=(1, 500, 1, 500), type='rectangle',
              meta=dict(file_path=ROOT_DIR + '/tests/test_data/five_blobs.txt', file_type=1))
    roi.to_yaml(path=path)

    dat = load_from_roi_file(path)
    assert(len(dat)==5)
