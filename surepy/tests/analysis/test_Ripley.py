import pytest
import numpy as np
import pandas as pd

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.analysis import Ripleys_k_function, Ripleys_l_function, Ripleys_h_function


@pytest.fixture()
def locdata_blobs():
    return io.load_txt_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/five_blobs.txt')

@pytest.fixture()
def locdata_blobs_3D():
    return io.load_txt_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/five_blobs_3D.txt')


def test_Ripleys_k_function(locdata_blobs):
    radii = np.linspace(0,100,20)
    rhf = Ripleys_k_function(locdata_blobs, radii=radii).compute()
    #print(rhf.results[0:5])
    assert (len(rhf.results) == len(radii))

def test_Ripleys_k_function_3d(locdata_blobs_3D):
    radii = np.linspace(0,100,20)
    rhf = Ripleys_k_function(locdata_blobs_3D, radii=radii).compute()
    #print(rhf.results[0:5])
    assert (len(rhf.results) == len(radii))

def test_Ripleys_k_function_estimate(locdata_blobs):
    radii = np.linspace(0,100,20)
    rhf = Ripleys_k_function(locdata_blobs, radii=radii, region_measure=1, other_locdata=locdata_blobs).compute()
    #print(rhf.results[0:5])
    assert (len(rhf.results) == len(radii))

def test_Ripleys_l_function(locdata_blobs):
    radii = np.linspace(0,100,20)
    rhf = Ripleys_l_function(locdata_blobs, radii=radii).compute()
    #print(rhf.results[0:5])
    assert (len(rhf.results) == len(radii))

def test_Ripleys_h_function(locdata_blobs):
    radii = np.linspace(0,100,20)
    rhf = Ripleys_h_function(locdata_blobs, radii=radii).compute()
    #print(rhf.results[0:5])
    assert (len(rhf.results) == len(radii))
