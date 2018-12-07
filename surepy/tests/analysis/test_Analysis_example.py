import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
from surepy.analysis import Analysis_example_algorithm_1
import surepy.tests.test_data

@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt', nrows=100)

def test_Analysis_example(locdata):
    #print(locdata.data.head())
    ae = Analysis_example_algorithm_1(locdata=None, limits=(100,110), meta={'comment': ' this is an example'})
    ae.compute()
    assert (len(list(ae.results)) == 10)

    # print(ae.results)
    # print(ae.meta)
    # print(ae.meta)
    # ae.plot()