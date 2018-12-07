import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
from surepy.analysis.localizations_per_frame import Localizations_per_frame, _DistributionFits
import surepy.tests.test_data


@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/rapidSTORM_dstorm_data.txt',
                                   nrows=100)


def test_Localizations_per_frame(locdata):
    # print(locdata.data.head())
    ana = Localizations_per_frame(locdata=locdata, norm=None, meta={'comment': 'this is an example'}).compute()
    # print(ana.results.head())
    assert(isinstance(ana.results, pd.Series))
    assert(ana.meta.comment == 'this is an example')
    print(ana.results.name)
    # ana.plot()

    ana = Localizations_per_frame(locdata=locdata, norm='Localization_density_bb').compute()
    assert(ana.results.name == 'number_localizations / Localization_density_bb')
    ana.fit_distributions(floc=0)
    assert(0 in ana.distribution_statistics.parameter_dict().values())
    # ana.plot()
    # ana.hist()


def test_Distribution_fits(locdata):
    ana = Localizations_per_frame(locdata=locdata).compute()
    distribution_statistics = _DistributionFits(ana)
    assert(distribution_statistics.parameter_dict()=={})
    # print(distribution_statistics)
    distribution_statistics.fit()
    assert(list(distribution_statistics.parameter_dict().keys()) ==
           ['number_localizations_center', 'number_localizations_sigma'])
    distribution_statistics.plot(show=False)
