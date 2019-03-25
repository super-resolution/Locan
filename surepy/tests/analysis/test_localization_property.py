import pytest
import pandas as pd
from scipy import stats

import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.analysis import LocalizationProperty
from surepy.analysis.localization_property import _DistributionFits


@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt',
                                   nrows=100)


def test_DistributionFits(locdata):
    lprop = LocalizationProperty(loc_property='intensity').compute(locdata=locdata)
    ds = _DistributionFits(lprop)
    ds.fit(distribution=stats.expon)
    # print(ds.parameters)
    # print(ds.__dict__)
    # #print(ds.parameter_dict())
    # ds.plot()
    assert(ds.parameters == ['intensity_loc', 'intensity_scale'])
    ds.fit(distribution=stats.expon, with_constraints=False, floc=1000)
    assert(ds.parameter_dict()['intensity_loc'] == 1000)


def test_Localization_property(locdata):
    lprop = LocalizationProperty(loc_property='intensity').compute(locdata=locdata)
    assert(lprop.results.columns == pd.Index(['intensity'], dtype='object'))
    assert(lprop.distribution_statistics is None)
    # lprop.plot()
    # lprop.hist()
    lprop.fit_distributions()
    assert(list(lprop.distribution_statistics.parameter_dict().keys()) == ['intensity_loc', 'intensity_scale'])

    lprop = LocalizationProperty(loc_property='intensity', index='frame').compute(locdata=locdata)
    # print(lprop.results)
    assert(lprop.results.index.name == 'frame')
    assert(lprop.results.columns == pd.Index(['intensity'], dtype='object'))
