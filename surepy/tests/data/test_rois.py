import pytest

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.data.rois import Roi, Roi_manager
from surepy.simulation import simulate_blobs


# fixtures

@pytest.fixture()
def locdata():
    dat = simulate_blobs(n_centers=10, n_samples=1000, n_features=2, center_box=(0, 1000), cluster_std=10, seed=0)
    return dat

# tests

def test_Roi(locdata):
    #print(locdata.meta)
    roi = Roi(reference=locdata, points=(1, 10, 1, 10), type='rectangle')
    assert(roi.reference is None)
    assert(roi._locdata is locdata)
    assert(roi.points==(1, 10, 1, 10))

    roi_2 = Roi(reference='my/path/to/file', points=(1, 10, 1, 10), type='rectangle')
    assert(roi_2.reference=='my/path/to/file')


def test_rois_io():
    roi = Roi(reference='my/path/to/file', points=(1, 10, 1, 10), type='polygon')

    path = ROOT_DIR + '/tests/test_data/roi.yaml'
    roi.save(path = path)

    roi_2 = Roi()
    assert(roi_2.type=='rectangle')
    roi_2.load(path = path)
    assert(roi_2.type=='polygon')


# def test_rois (locdata):
#     roim = Roi_manager()
#     roim.add_rectangle((1,10,1,10))
#     roim.reference = locdata
#     assert (roim.rois==[{'points': (1, 10, 1, 10), 'type': 'rectangle'}])
#     roim.clear()
#     assert (roim.rois==[])
#     roim.clear()
#     roim.add_rectangles([(1,500,1,500),(500,1000,500,1000)])
#     assert (roim.rois==[{'points': (1, 500,1,500), 'type': 'rectangle'},
#                         {'points': (500,1000,500,1000), 'type': 'rectangle'}])
#     assert(isinstance(roim.locdata, LocData))
#     assert(len(roim.locdatas)==2)
#     #roim.select_by_drawing(locdata)
#
#     #print(roim.rois)
#

