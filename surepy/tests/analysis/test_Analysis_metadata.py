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


def test_Localization_precision(locdata):
    # print(locdata.data.head())
    lp = Localization_precision(locdata=locdata, meta={'comment': 'my comment'})
    #print(lp.meta)
    assert (lp.meta.method.name == "Localization_precision")
    assert (lp.meta.comment == 'my comment')
