import pytest
import numpy as np
import pandas as pd
from scipy import stats

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.analysis import Localization_property
from surepy.analysis.localization_property import _Distribution_stats


@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt', nrows=100)


def test_Distribution_stats(locdata):
    lprop = Localization_property(locdata=locdata, loc_property='Intensity').compute()
    ds = _Distribution_stats(lprop)
    ds.fit(distribution=stats.expon)
    #ds.plot()
    assert(ds.parameters == ['loc', 'scale'])

def test_Localization_property(locdata):
    lprop = Localization_property(locdata=locdata, loc_property='Intensity').compute()
    assert(lprop.results.columns == pd.Index(['Intensity'], dtype='object'))
    assert(lprop.distribution_statistics is None)
    #lprop.plot()
    #lprop.hist()
    lprop.fit_distributions()
    assert(list(lprop.distribution_statistics.parameter_dict().keys()) == ['loc', 'scale'])

    lprop = Localization_property(locdata=locdata, loc_property='Intensity', index='Frame').compute()
    # print(lprop.results)
    assert(lprop.results.index.name == 'Frame')
    assert(lprop.results.columns == pd.Index(['Intensity'], dtype='object'))
