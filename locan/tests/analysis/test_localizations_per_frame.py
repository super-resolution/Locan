from copy import deepcopy

import pytest
import pandas as pd
import matplotlib.pyplot as plt

from locan import LocData, LocalizationsPerFrame
from locan.analysis.localizations_per_frame import _localizations_per_frame, _DistributionFits


def test__localizations_per_frame(caplog, locdata_rapidSTORM_2d):
    series = _localizations_per_frame(locdata=locdata_rapidSTORM_2d)
    assert len(series) == 48
    assert series.iloc[0] == 22
    assert series.index[-1] == 47
    assert series.name == "n_localizations"

    series = _localizations_per_frame(locdata=locdata_rapidSTORM_2d, norm=2, time_delta=None)
    assert len(series) == 48
    assert series.iloc[0] == 11
    assert series.index[-1] == 47
    assert series.name == "n_localizations / 2"

    series = _localizations_per_frame(locdata=locdata_rapidSTORM_2d, norm=2)
    assert len(series) == 48
    assert series.iloc[0] == 11
    assert series.index[-1] == 47

    series = _localizations_per_frame(locdata=locdata_rapidSTORM_2d, time_delta=10)
    assert len(series) == 48
    assert series.iloc[0] == 22
    assert series.index.seconds[-1] == 47 * 10
    assert series.name == "n_localizations / s"

    series = _localizations_per_frame(locdata=range(10))
    assert len(series) == 10
    assert series.iloc[0] == 1
    assert series.index[-1] == 9
    assert series.name == "n_localizations"

    series = _localizations_per_frame(locdata=locdata_rapidSTORM_2d, time_delta=10, resample="10s")
    assert len(series) == 48
    assert series.iloc[0] == 22
    assert series.index.seconds[-1] == 47 * 10
    assert series.name == "n_localizations / s"

    with pytest.raises(TypeError):
        _localizations_per_frame(locdata=range(10), resample="2s")

    assert caplog.record_tuples == [('locan.analysis.localizations_per_frame', 30,
                                     'integration_time not available in locdata.meta - frames used instead.'),
                                    ('locan.analysis.localizations_per_frame', 30,
                                     'integration_time not available in locdata.meta - frames used instead.'),
                                    ('locan.analysis.localizations_per_frame', 30,
                                     'integration_time not available in locdata.meta - frames used instead.'),
                                    ('locan.analysis.localizations_per_frame', 30,
                                     'integration_time not available in locdata.meta - frames used instead.')
                                    ]


def test_Localizations_per_frame_empty(caplog):
    lpf = LocalizationsPerFrame().compute(LocData())
    lpf.fit_distributions()
    lpf.plot()
    lpf.hist()
    assert caplog.record_tuples == [('locan.analysis.localizations_per_frame', 30, 'Locdata is empty.'),
                                    ('locan.analysis.localizations_per_frame', 30, 'No results available to fit.')]


def test_Localizations_per_frame(locdata_rapidSTORM_2d):
    lpf = LocalizationsPerFrame(meta={'comment': 'this is an example'}).compute(locdata=locdata_rapidSTORM_2d)
    # print(lpf.results.head())
    assert(isinstance(lpf.results, pd.Series))
    assert len(lpf.results) == 48
    assert lpf.results.iloc[0] == 22
    assert lpf.results.index[-1] == 47
    assert(lpf.meta.comment == 'this is an example')

    assert lpf.accumulation_time() == 22
    assert lpf.accumulation_time(fraction=0.8) == 38

    lpf.plot()
    lpf.plot(window=10)
    lpf.hist()
    # plt.show()

    lpf = LocalizationsPerFrame(norm='localization_density_bb', time_delta=1, resample="2s")\
        .compute(locdata=locdata_rapidSTORM_2d)
    assert(lpf.results.name == 'n_localizations / localization_density_bb / s')

    lpf.plot()
    lpf.plot(window=10)
    lpf.hist()
    # plt.show()

    lpf.fit_distributions(floc=0)
    assert(0 in lpf.distribution_statistics.parameter_dict().values())

    lpf = LocalizationsPerFrame(norm=5).compute(locdata=locdata_rapidSTORM_2d)
    assert(lpf.results.name == 'n_localizations / 5')

    locdata_rapidSTORM_2d_ = deepcopy(locdata_rapidSTORM_2d)
    locdata_rapidSTORM_2d_.meta.experiment.setups.add().optical_units.add().detection.camera.integration_time = 10E-3
    lpf = LocalizationsPerFrame(norm=None).compute(locdata=locdata_rapidSTORM_2d_)
    assert(lpf.results.name == 'n_localizations / s')

    plt.close('all')


def test_Distribution_fits(locdata_rapidSTORM_2d):
    lpf = LocalizationsPerFrame().compute(locdata=locdata_rapidSTORM_2d)
    distribution_statistics = _DistributionFits(lpf)
    assert distribution_statistics.parameter_dict() == {}
    # print(distribution_statistics)
    distribution_statistics.fit()
    assert(list(distribution_statistics.parameter_dict().keys()) ==
           ['n_localizations_center', 'n_localizations_sigma'])
    distribution_statistics.plot()


@pytest.mark.visual
def test_Localizations_per_frame_cumulative(locdata_rapidSTORM_2d):
    lpf = LocalizationsPerFrame().compute(locdata=locdata_rapidSTORM_2d)

    lpf.plot()
    plt.show()

    lpf.plot(cumulative=True)
    plt.show()

    lpf.plot(cumulative=True, normalize=True)
    plt.show()

    lpf.hist(density=False)
    plt.show()

    lpf.hist(density=True, cumulative=True)
    plt.show()

    plt.close('all')
