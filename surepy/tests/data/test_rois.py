import pytest
import matplotlib.pyplot as plt
from surepy.data.rois import Roi_manager
from surepy.simulation import simulate_blobs
from surepy.render import render2D

# fixtures

@pytest.fixture()
def locdata():
    dat = simulate_blobs(n_centers=10, n_samples=1000, n_features=2, center_box=(0, 1000), cluster_std=10, seed=0)
    return dat

# tests

def test_rois (locdata):
    roim = Roi_manager()
    roim.add_rectangle((1,10,1,10))
    assert (roim.rois==[{'points': (1, 10, 1, 10), 'type': 'rectangle'}])
    roim.clear()
    assert (roim.rois==[])
    roim.clear()
    roim.add_rectangles([(1,100,1,100),(100,200,100,200)])
    assert (roim.rois==[{'points': (1, 100, 1, 100), 'type': 'rectangle'},
                        {'points': (100, 200, 100, 200), 'type': 'rectangle'}])
    #roim.select_by_drawing(locdata)

    #print(roim.rois)

