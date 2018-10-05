import pytest

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.data.rois import Roi_manager
from surepy.simulation import simulate_blobs


# fixtures

@pytest.fixture()
def locdata():
    dat = simulate_blobs(n_centers=10, n_samples=1000, n_features=2, center_box=(0, 1000), cluster_std=10, seed=0)
    return dat

# tests

def test_rois (locdata):
    roim = Roi_manager()
    roim.add_rectangle((1,10,1,10))
    roim.reference = locdata
    assert (roim.rois==[{'points': (1, 10, 1, 10), 'type': 'rectangle'}])
    roim.clear()
    assert (roim.rois==[])
    roim.clear()
    roim.add_rectangles([(1,500,1,500),(500,1000,500,1000)])
    assert (roim.rois==[{'points': (1, 500,1,500), 'type': 'rectangle'},
                        {'points': (500,1000,500,1000), 'type': 'rectangle'}])
    assert(isinstance(roim.locdata, LocData))
    assert(len(roim.locdatas)==2)
    #roim.select_by_drawing(locdata)

    #print(roim.rois)

def test_rois_io():
    roim = Roi_manager()
    roim.add_rectangle([1,10,1,10])

    path = ROOT_DIR + '/tests/test_data/rois.yaml'
    roim.save(path = path)

    roim_2 = Roi_manager()
    roim_2.load(path = path)
    assert(roim.rois == roim_2.rois)
    #print(roim_2.rois)

