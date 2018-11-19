import pytest
import pandas as pd

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.data.rois import Roi


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
    #print(locdata.meta)
    roi = Roi(points=(1, 10, 1, 10), type='rectangle')
    assert(roi.reference is None)
    assert(roi._locdata is None)
    assert(roi.points==(1, 10, 1, 10))

    roi = Roi(reference=locdata, points=(1, 10, 1, 10), type='rectangle')
    assert(roi.reference == '')
    assert(roi._locdata is locdata)
    assert(roi.points==(1, 10, 1, 10))

    roi = Roi(reference='my/path/to/file', points=(1, 10, 1, 10), type='rectangle')
    assert(roi.reference=='my/path/to/file')
    assert(roi._locdata is None)

    roi = Roi(reference=locdata, points=(0,3), type='rectangle')
    dat_1 = roi.locdata()
    assert(len(dat_1)==6)
    roi = Roi(reference=locdata, points=(0,3,0,3), type='rectangle')
    dat_1 = roi.locdata()
    assert(len(dat_1)==5)
    roi = Roi(reference=locdata, points=(0,3,0,3,0,3), type='rectangle')
    dat_1 = roi.locdata()
    assert(len(dat_1)==4)


def test_rois_io():
    roi = Roi(reference='my/path/to/file', points=(1, 10, 1, 10), type='polygon')

    path = ROOT_DIR + '/tests/test_data/roi.yaml'
    roi.to_yaml(path = path)

    roi_2 = Roi()
    assert(roi_2.type=='rectangle')
    roi_2.from_yaml(path = path)
    assert(roi_2.type=='polygon')

