import pytest
import numpy as np
import pandas as pd
from scipy import stats

from surepy import LocData
from surepy.analysis.blinking import _blink_statistics, _DistributionFits
from surepy.analysis import BlinkStatistics


@pytest.fixture()
def locdata_simple():
    locdata_dict = {
        'position_x': [0, 0, 1, 4, 5],
        'position_y': [0, 1, 3, 4, 1],
        'intensity': [0, 1, 3, 4, 1],
        'psf_sigma_x': [100, 100, 100, 100, 100],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


@pytest.fixture()
def locdata_with_zero_frame():
    locdata_dict = {
        'frame': [0, 1, 2, 4, 10, 11, 14]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


@pytest.fixture()
def locdata_without_zero_frame():
    locdata_dict = {
        'frame': [1, 2, 4, 10, 11, 14]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


@pytest.fixture()
def locdata_with_repetitions():
    locdata_dict = {
        'frame': [2, 2, 2, 4, 4, 14]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


def test_blink_statistics(locdata_with_zero_frame, locdata_without_zero_frame):
    bs = _blink_statistics(locdata_with_zero_frame, memory=0, remove_heading_off_periods=False)
    assert all(bs['on_periods'] == [3, 1, 2, 1])
    assert all(bs['off_periods'] == [1, 5, 2])

    bs = _blink_statistics(locdata_with_zero_frame.data.frame.values, memory=0, remove_heading_off_periods=False)
    assert all(bs['on_periods'] == [3, 1, 2, 1])
    assert all(bs['off_periods'] == [1, 5, 2])

    bs = _blink_statistics(locdata_without_zero_frame, memory=0, remove_heading_off_periods=False)
    assert all(bs['on_periods'] == [2, 1, 2, 1])
    assert all(bs['off_periods'] == [1, 1, 5, 2])

    bs = _blink_statistics(locdata_with_zero_frame, memory=0, remove_heading_off_periods=True)
    assert all(bs['on_periods'] == [3, 1, 2, 1])
    assert all(bs['off_periods'] == [1, 5, 2])

    bs = _blink_statistics(locdata_without_zero_frame, memory=0, remove_heading_off_periods=True)
    assert all(bs['on_periods'] == [2, 1, 2, 1])
    assert all(bs['off_periods'] == [1, 5, 2])

    bs = _blink_statistics(locdata_with_zero_frame, memory=1, remove_heading_off_periods=False)
    assert all(bs['on_periods'] == [5, 2, 1])
    assert all(bs['off_periods'] == [5, 2])

    bs = _blink_statistics(locdata_without_zero_frame, memory=1, remove_heading_off_periods=False)
    assert all(bs['on_periods'] == [4, 2, 1])
    assert all(bs['off_periods'] == [1, 5, 2])

    bs = _blink_statistics(locdata_with_zero_frame, memory=2, remove_heading_off_periods=False)
    assert all(bs['on_periods'] == [5, 5])
    assert all(bs['off_periods'] == [5])

    bs = _blink_statistics(locdata_without_zero_frame, memory=2, remove_heading_off_periods=False)
    assert all(bs['on_periods'] == [4, 5])
    assert all(bs['off_periods'] == [1, 5])


def test_blink_statistics__with_repetitions(locdata_with_repetitions):
    with pytest.warns(UserWarning):
        _blink_statistics(locdata_with_repetitions, memory=0, remove_heading_off_periods=False)


def test_BlinkStatistics(locdata_with_zero_frame):
    bs = BlinkStatistics().compute(locdata_with_zero_frame)
    assert repr(bs) == "BlinkStatistics(**{'memory': 0, 'remove_heading_off_periods': True})"
    assert all(bs.results['on_periods'] == [3, 1, 2, 1])
    assert all(bs.results['off_periods'] == [1, 5, 2])

    bs.hist(data_identifier='on_periods', ax=None, show=False, bins='auto', log=True, fit=False)
    bs.hist(data_identifier='off_periods', ax=None, show=False, bins='auto', log=True, fit=False)


def test_DistributionFits(locdata_with_zero_frame):
    bs = BlinkStatistics().compute(locdata_with_zero_frame)
    df = _DistributionFits(bs, distribution=stats.expon, data_identifier='on_periods')
    assert repr(df) == "_DistributionFits(analysis_class=BlinkStatistics, " \
                       "distribution=expon_gen, data_identifier=on_periods)"
    assert df.parameter_dict() == {}
    df.fit()
    assert list(df.parameter_dict().keys()) == ['on_periods_loc', 'on_periods_scale']

    df = _DistributionFits(bs, distribution=stats.expon, data_identifier='off_periods')
    df.fit()
    assert list(df.parameter_dict().keys()) == ['off_periods_loc', 'off_periods_scale']
    df.plot(show=False)


def test_fit_distributions(locdata_with_zero_frame):
    bs = BlinkStatistics().compute(locdata_with_zero_frame)
    bs.fit_distributions()
    assert bs.distribution_statistics['on_periods'].parameter_dict() == \
           {'on_periods_loc': 1.0, 'on_periods_scale': 0.75}
    assert bs.distribution_statistics['off_periods'].parameter_dict() == \
           {'off_periods_loc': 1.0, 'off_periods_scale': 1.6666666666666665}
    bs.hist(show=False)
    bs.hist(data_identifier='off_periods', show=False)

