import pytest
import numpy as np
import pandas as pd

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.analysis import Localization_property


@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt', nrows=100)


def test_Localization_property(locdata):
    lprop = Localization_property(locdata=locdata, property='Intensity', index='Frame')
    # print(lprop.results)
    assert(lprop.results.index.name == 'Frame')
    assert(lprop.results.columns == pd.Index(['Intensity'], dtype='object'))
