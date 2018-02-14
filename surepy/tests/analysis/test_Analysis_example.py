import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.analysis import Analysis_example

@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt', nrows=100)

def test_Analysis_example(locdata):
    #print(locdata.data.head())
    ae = Analysis_example(locdata=None, param=1)
    #print(ae.results)
    assert (len(list(ae.results)) == 2)

