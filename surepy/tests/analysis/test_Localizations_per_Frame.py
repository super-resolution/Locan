import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
from surepy.analysis.localizations_per_frame import Localizations_per_frame
import surepy.tests.test_data

@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt', nrows=100)

def test_Localizations_per_frame(locdata):
    #print(locdata.data.head())
    ana = Localizations_per_frame(locdata=locdata, norm=None, meta={'comment': 'this is an example'})
    assert(ana.meta.comment == 'this is an example')

    ana.compute()
    assert(isinstance(ana.results, pd.Series))
    #print(ana.results.head())