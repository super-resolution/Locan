import pytest
import numpy as np
import pandas as pd

from surepy import LocData
from surepy.analysis.blinking import _blink_statistics
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
    assert all(bs[0] == [3, 1, 2, 1])
    assert all(bs[1] == [1, 5, 2])

    bs = _blink_statistics(locdata_with_zero_frame.data.frame.values, memory=0, remove_heading_off_periods=False)
    assert all(bs[0] == [3, 1, 2, 1])
    assert all(bs[1] == [1, 5, 2])

    bs = _blink_statistics(locdata_without_zero_frame, memory=0, remove_heading_off_periods=False)
    assert all(bs[0] == [2, 1, 2, 1])
    assert all(bs[1] == [1, 1, 5, 2])

    bs = _blink_statistics(locdata_with_zero_frame, memory=0, remove_heading_off_periods=True)
    assert all(bs[0] == [3, 1, 2, 1])
    assert all(bs[1] == [1, 5, 2])

    bs = _blink_statistics(locdata_without_zero_frame, memory=0, remove_heading_off_periods=True)
    assert all(bs[0] == [2, 1, 2, 1])
    assert all(bs[1] == [1, 5, 2])

    bs = _blink_statistics(locdata_with_zero_frame, memory=1, remove_heading_off_periods=False)
    assert all(bs[0] == [5, 2, 1])
    assert all(bs[1] == [5, 2])

    bs = _blink_statistics(locdata_without_zero_frame, memory=1, remove_heading_off_periods=False)
    assert all(bs[0] == [4, 2, 1])
    assert all(bs[1] == [1, 5, 2])

    bs = _blink_statistics(locdata_with_zero_frame, memory=2, remove_heading_off_periods=False)
    assert all(bs[0] == [5, 5])
    assert all(bs[1] == [5])

    bs = _blink_statistics(locdata_without_zero_frame, memory=2, remove_heading_off_periods=False)
    assert all(bs[0] == [4, 5])
    assert all(bs[1] == [1, 5])


def test_blink_statistics__with_repetitions(locdata_with_repetitions):
    with pytest.warns(UserWarning):
        _blink_statistics(locdata_with_repetitions, memory=0, remove_heading_off_periods=False)


def test_BlinkStatistics(locdata_with_zero_frame):
    bs = BlinkStatistics().compute(locdata_with_zero_frame)
    assert repr(bs) == "BlinkStatistics(**{'memory': 0, 'remove_heading_off_periods': True})"
    assert all(bs.results[0] == [3, 1, 2, 1])
    assert all(bs.results[1] == [1, 5, 2])
