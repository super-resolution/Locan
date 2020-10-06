import pytest
import pandas as pd

import surepy.constants
import surepy.io.io_locdata as io
from surepy.analysis.localizations_per_frame import LocalizationsPerFrame, _DistributionFits


@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
                                   nrows=100)


def test_Localizations_per_frame(locdata):
    # print(locdata.data.head())
    ana = LocalizationsPerFrame(norm=None, meta={'comment': 'this is an example'}).compute(locdata=locdata)
    # print(ana.results.head())
    assert(isinstance(ana.results, pd.Series))
    assert(ana.meta.comment == 'this is an example')
    # print(ana.results.name)
    # ana.plot()

    ana = LocalizationsPerFrame(norm='localization_density_bb').compute(locdata=locdata)
    assert(ana.results.name == 'n_localizations / localization_density_bb')
    ana.fit_distributions(floc=0)
    assert(0 in ana.distribution_statistics.parameter_dict().values())
    # ana.plot()
    # ana.hist()


def test_Distribution_fits(locdata):
    ana = LocalizationsPerFrame().compute(locdata=locdata)
    distribution_statistics = _DistributionFits(ana)
    assert distribution_statistics.parameter_dict() == {}
    # print(distribution_statistics)
    distribution_statistics.fit()
    assert(list(distribution_statistics.parameter_dict().keys()) ==
           ['n_localizations_center', 'n_localizations_sigma'])
    distribution_statistics.plot()
