import pytest
import matplotlib.pyplot as plt  # needed for visual inspection

from locan import LocData
from locan.analysis import LocalizationProperty2d


def test_Localization_property_2d_empty(caplog):
    lprop = LocalizationProperty2d().compute(LocData())
    lprop.plot()
    lprop.plot_residuals()
    lprop.plot_deviation_from_mean()
    lprop.plot_deviation_from_median()
    lprop.report()
    assert caplog.record_tuples == [('locan.analysis.localization_property_2d', 30, 'Locdata is empty.'),
                                    ('locan.analysis.localization_property_2d', 30, 'No results available')]


def test_Localization_property_2d(capfd, locdata_rapidSTORM_2d):
    lprop = LocalizationProperty2d(meta=None, other_property='local_background', bin_size=1000)\
        .compute(locdata_rapidSTORM_2d)
    assert 'model_result' in lprop.results._fields
    assert lprop.results.model_result.params

    lprop.report()
    captured = capfd.readouterr()
    assert captured.out[:16] == "Fit results for:"

    lprop.plot()
    lprop.plot_residuals()
    lprop.plot_deviation_from_mean()
    lprop.plot_deviation_from_median()
    # plt.show()
