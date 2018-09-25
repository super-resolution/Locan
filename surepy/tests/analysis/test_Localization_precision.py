import pytest
import numpy as np
import pandas as pd

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.analysis import Localization_precision


@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt', nrows=100)

@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': range(10),
        'Position_y': range(10),
        'Frame': [1,1,1,2,2,3,5,8,9,10]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


def test_Localization_precision(locdata):
    # print(locdata.data.head())
    lp = Localization_precision(locdata=locdata)
    lp.compute()
    # print(lp.results)
    assert (len(lp.results) == 33)

def test_Localization_precision_2(locdata_simple):
    print(locdata_simple.data)
    lp = Localization_precision(locdata=locdata_simple)
    lp.compute()
    print(lp.results)
    #assert (len(lp.results) == 33)

