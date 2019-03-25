import pytest
import numpy as np
import pandas as pd

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.analysis import RipleysKFunction, RipleysLFunction, RipleysHFunction


@pytest.fixture()
def locdata_blobs():
    return io.load_txt_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/five_blobs.txt')


@pytest.fixture()
def locdata_blobs_3d():
    return io.load_txt_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/five_blobs_3D.txt')


def test_Ripleys_k_function(locdata_blobs):
    radii = np.linspace(0, 100, 20)
    rhf = RipleysKFunction(radii=radii).compute(locdata_blobs)
    # print(rhf.results[0:5])
    assert (len(rhf.results) == len(radii))
    # rhf.plot()


def test_Ripleys_k_function_3d(locdata_blobs_3d):
    radii = np.linspace(0, 100, 20)
    rhf = RipleysKFunction(radii=radii).compute(locdata_blobs_3d)
    # print(rhf.results[0:5])
    assert (len(rhf.results) == len(radii))


def test_Ripleys_k_function_estimate(locdata_blobs):
    radii = np.linspace(0, 100, 20)
    rhf = RipleysKFunction(radii=radii, region_measure=1).compute(locdata_blobs, other_locdata=locdata_blobs)
    # print(rhf.results[0:5])
    assert (len(rhf.results) == len(radii))


def test_Ripleys_l_function(locdata_blobs):
    radii = np.linspace(0, 100, 20)
    rhf = RipleysLFunction(radii=radii).compute(locdata_blobs)
    # print(rhf.results[0:5])
    assert (len(rhf.results) == len(radii))


def test_Ripleys_h_function(locdata_blobs):
    radii = np.linspace(0, 100, 20)
    rhf = RipleysHFunction(radii=radii).compute(locdata_blobs)
    # print(rhf._Ripley_h_maximum)
    # print(rhf.Ripley_h_maximum)
    # print(rhf.results[5:11])
    assert len(rhf.results) == len(radii)
    assert len(rhf.Ripley_h_maximum) == 2
    del rhf.Ripley_h_maximum
    assert len(rhf.Ripley_h_maximum) == 2
    # rhf.plot()
