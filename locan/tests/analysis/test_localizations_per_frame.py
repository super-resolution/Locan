import pytest
import pandas as pd

from locan import LocData
from locan.analysis.localizations_per_frame import LocalizationsPerFrame, _DistributionFits


def test_Localizations_per_frame_empty(caplog):
    lpf = LocalizationsPerFrame().compute(LocData())
    lpf.fit_distributions()
    lpf.plot()
    lpf.hist()
    assert caplog.record_tuples == [('locan.analysis.localizations_per_frame', 30, 'Locdata is empty.'),
                                    ('locan.analysis.localizations_per_frame', 30, 'No results available to fit.')]


def test_Localizations_per_frame(locdata_rapidSTORM_2d):
    # print(locdata.data.head())
    ana = LocalizationsPerFrame(norm=None, meta={'comment': 'this is an example'}).compute(locdata=locdata_rapidSTORM_2d)
    # print(ana.results.head())
    assert(isinstance(ana.results, pd.Series))
    assert(ana.meta.comment == 'this is an example')
    # print(ana.results.name)
    ana.plot()
    ana.hist()

    ana = LocalizationsPerFrame(norm='localization_density_bb').compute(locdata=locdata_rapidSTORM_2d)
    assert(ana.results.name == 'n_localizations / localization_density_bb')
    ana.fit_distributions(floc=0)
    assert(0 in ana.distribution_statistics.parameter_dict().values())

    ana = LocalizationsPerFrame(norm=5).compute(locdata=locdata_rapidSTORM_2d)
    assert(ana.results.name == 'n_localizations / 5')


def test_Distribution_fits(locdata_rapidSTORM_2d):
    ana = LocalizationsPerFrame().compute(locdata=locdata_rapidSTORM_2d)
    distribution_statistics = _DistributionFits(ana)
    assert distribution_statistics.parameter_dict() == {}
    # print(distribution_statistics)
    distribution_statistics.fit()
    assert(list(distribution_statistics.parameter_dict().keys()) ==
           ['n_localizations_center', 'n_localizations_sigma'])
    distribution_statistics.plot()
