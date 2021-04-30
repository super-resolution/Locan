from collections import namedtuple

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from locan import LocData
from locan.analysis.blinking import _blink_statistics, _DistributionFits
from locan.analysis import BlinkStatistics


# frame with on and off periods up to three frames and starting with two-frame on-period.
FramesTest = namedtuple('FramesTest',
                         ['frames',
                          'on_periods_expected', 'on_periods_frame_expected',
                          'off_periods_expected', 'off_periods_frame_expected'])

@pytest.fixture()
def frames_1():
    return FramesTest(
        frames=np.array([0, 4, 6, 7, 8, 12, 13]),
        on_periods_expected=[1, 1, 3, 2],
        on_periods_frame_expected=[0, 4, 6, 12],
        off_periods_expected=[3, 1, 3],
        off_periods_frame_expected=[1, 5, 9]
        )

# frame with on and off periods up to three frames and starting with two-frame on-period.
@pytest.fixture()
def frames_2():
    return FramesTest(
        frames=np.array([0, 1, 4, 6, 7, 8, 12, 13]),
        on_periods_expected=[2, 1, 3, 2],
        on_periods_frame_expected=[0, 4, 6, 12],
        off_periods_expected=[2, 1, 3],
        off_periods_frame_expected=[2, 5, 9]
        )

# frame with on and off periods up to three frames and starting with off-period.
@pytest.fixture()
def frames_3():
    return FramesTest(
        frames=np.array([0, 1, 4, 6, 7, 8, 12, 13]) + 4,
        on_periods_expected=[2, 1, 3, 2],
        on_periods_frame_expected=[4, 8, 10, 16],
        off_periods_expected=[4, 2, 1, 3],
        off_periods_frame_expected=[0, 6, 9, 13]
        )

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


@pytest.mark.parametrize('fixture_name, expected', [
    ('frames_1', 0),
    ('frames_2', 0),
    ('frames_3', 0),
])
def test__blink_statistics(frames_1, frames_2, frames_3, fixture_name, expected):
    frames_ = eval(fixture_name)
    results = _blink_statistics(frames_.frames, memory=0, remove_heading_off_periods=False)
    assert len(results['on_periods']) == len(results['on_periods_frame'])
    assert len(results['off_periods']) == len(results['off_periods_frame'])
    assert np.array_equal(results['on_periods'], frames_.on_periods_expected)
    assert np.array_equal(results['off_periods'], frames_.off_periods_expected)
    assert np.array_equal(results['on_periods_frame'], frames_.on_periods_frame_expected)
    assert np.array_equal(results['off_periods_frame'], frames_.off_periods_frame_expected)
    for op, op_indices in zip(results['on_periods'], results['on_periods_indices']):
        assert len(op_indices) == op
    assert np.sum([len(op_indices) for op_indices in results['on_periods_indices']]) == len(frames_.frames)


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
    _blink_statistics(locdata_with_repetitions, memory=0, remove_heading_off_periods=False)


def test_BlinkStatistics_empty(caplog):
    bs = BlinkStatistics().compute(LocData())
    bs.fit_distributions()
    bs.hist()
    assert caplog.record_tuples == [('locan.analysis.blinking', 30, 'Locdata is empty.'),
                                    ('locan.analysis.blinking', 30, 'No results available to fit.')]


def test_BlinkStatistics(locdata_with_zero_frame):
    bs = BlinkStatistics().compute(locdata_with_zero_frame)
    assert repr(bs) == "BlinkStatistics(memory=0, remove_heading_off_periods=True)"
    assert all(bs.results['on_periods'] == [3, 1, 2, 1])
    assert all(bs.results['off_periods'] == [1, 5, 2])
    assert bs.distribution_statistics == {}

    bs.hist(data_identifier='on_periods', ax=None, bins='auto', log=True, fit=False)
    bs.hist(data_identifier='off_periods', ax=None, bins='auto', log=True, fit=False)
    bs.hist(data_identifier='on_periods', ax=None, bins='auto', log=True, fit=True)


def test_DistributionFits(locdata_with_zero_frame):
    bs = BlinkStatistics().compute(locdata_with_zero_frame)
    df = _DistributionFits(bs, distribution=stats.expon, data_identifier='on_periods')
    # print(df.analysis_class.results)
    assert len(df.analysis_class.results) == 5
    assert df.data_identifier == 'on_periods'
    assert repr(df) == "_DistributionFits(analysis_class=BlinkStatistics, " \
                       "distribution=expon_gen, data_identifier=on_periods)"
    assert df.parameter_dict() == {}
    df.fit()
    assert list(df.parameter_dict().keys()) == ['on_periods_loc', 'on_periods_scale']

    df = _DistributionFits(bs, distribution=stats.expon, data_identifier='off_periods')
    df.fit()
    assert list(df.parameter_dict().keys()) == ['off_periods_loc', 'off_periods_scale']
    df.plot()

    # print(df.analysis_class.results[df.data_identifier])


def test_fit_distributions(locdata_with_zero_frame):
    bs = BlinkStatistics().compute(locdata_with_zero_frame)
    bs.fit_distributions()
    assert bs.distribution_statistics['on_periods'].parameter_dict() == \
           {'on_periods_loc': 1.0, 'on_periods_scale': 0.75}
    assert bs.distribution_statistics['off_periods'].parameter_dict() == \
           {'off_periods_loc': 1.0, 'off_periods_scale': 1.6666666666666665}
    bs.hist()
    bs.hist(data_identifier='off_periods')
    del bs

    bs = BlinkStatistics().compute(locdata_with_zero_frame)
    bs.fit_distributions(with_constraints=False)
    assert bs.distribution_statistics['on_periods'].parameter_dict()['on_periods_loc'] == 1
    assert bs.distribution_statistics['off_periods'].parameter_dict()['off_periods_loc'] == 1
    del bs

    bs = BlinkStatistics().compute(locdata_with_zero_frame)
    bs.fit_distributions(data_identifier='on_periods')
    assert bs.distribution_statistics['on_periods'].parameter_dict() == \
           {'on_periods_loc': 1.0, 'on_periods_scale': 0.75}
