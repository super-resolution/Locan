import pytest
import numpy as np
import pandas as pd

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.analysis import Ripleys_h_function


@pytest.fixture()
def locdata_blobs():
    return io.load_txt_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/five_blobs.txt')


def test_Ripleys_k_function(locdata_blobs):
    radii = np.linspace(0,100,20)
    rhf = Ripleys_h_function(locdata_blobs, radii=radii).compute()
    #print(rhf.results)
    assert (len(rhf.results) == len(radii))
