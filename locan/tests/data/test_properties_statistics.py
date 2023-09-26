import numpy as np
import pandas as pd
import pytest

from locan import LocData
from locan.data.properties import range_from_collection, ranges, statistics


@pytest.fixture()
def locdata_simple():
    dict_ = {"position_x": [0, 0, 1, 4, 5], "position_y": [0, 1, 3, 4, 1]}
    return LocData(dataframe=pd.DataFrame.from_dict(dict_))


def test_statistics(locdata_simple):
    stat = statistics(locdata=locdata_simple)
    assert stat == {
        "position_x_count": 5.0,
        "position_x_min": 0.0,
        "position_x_max": 5.0,
        "position_x_mean": 2.0,
        "position_x_median": 1.0,
        "position_x_std": 2.3452078799117149,
        "position_x_sem": 1.0488088481701516,
        "position_y_count": 5.0,
        "position_y_min": 0.0,
        "position_y_max": 4.0,
        "position_y_mean": 1.8,
        "position_y_median": 1.0,
        "position_y_std": 1.6431676725154984,
        "position_y_sem": 0.73484692283495345,
    }


def test_statistics_from_dataframe(locdata_simple):
    stat = statistics(locdata=locdata_simple.data)
    assert stat == {
        "position_x_count": 5.0,
        "position_x_min": 0.0,
        "position_x_max": 5.0,
        "position_x_mean": 2.0,
        "position_x_median": 1.0,
        "position_x_std": 2.3452078799117149,
        "position_x_sem": 1.0488088481701516,
        "position_y_count": 5.0,
        "position_y_min": 0.0,
        "position_y_max": 4.0,
        "position_y_mean": 1.8,
        "position_y_median": 1.0,
        "position_y_std": 1.6431676725154984,
        "position_y_sem": 0.73484692283495345,
    }


def test_statistics_with_one_statfunction(locdata_simple):
    stat = statistics(locdata=locdata_simple, statistic_keys="min")
    assert stat == {"position_x_min": 0, "position_y_min": 0}


def test_statistics_from_Series(locdata_simple):
    stat = statistics(locdata=locdata_simple.data["position_x"], statistic_keys="mean")
    assert stat["position_x_mean"] == 2
    stat = statistics(
        locdata=locdata_simple.data["position_x"], statistic_keys=("min", "max")
    )
    assert stat == {"position_x_min": 0, "position_x_max": 5}


def test_statistics_with_new_column(locdata_simple):
    locdata_simple.dataframe = locdata_simple.dataframe.assign(new=np.arange(5))
    stat = statistics(locdata=locdata_simple)
    # print(stat)
    assert stat == {
        "position_x_count": 5.0,
        "position_x_min": 0.0,
        "position_x_max": 5.0,
        "position_x_mean": 2.0,
        "position_x_median": 1.0,
        "position_x_std": 2.3452078799117149,
        "position_x_sem": 1.0488088481701516,
        "position_y_count": 5.0,
        "position_y_min": 0.0,
        "position_y_max": 4.0,
        "position_y_mean": 1.8,
        "position_y_median": 1.0,
        "position_y_std": 1.6431676725154984,
        "position_y_sem": 0.73484692283495345,
        "new_count": 5.0,
        "new_min": 0.0,
        "new_max": 4.0,
        "new_mean": 2.0,
        "new_median": 2.0,
        "new_std": 1.5811388300841898,
        "new_sem": 0.70710678118654757,
    }


def test_range_from_collection(locdata_3d):
    collection = LocData.from_chunks(locdata_3d, n_chunks=2)
    result = range_from_collection(collection.references)
    assert result.position_x.min == 1
    assert result.position_x.max == 5
    assert result.position_z.min == 1
    assert result.position_z.max == 5

    result = range_from_collection(collection.references, loc_properties=["position_x"])
    assert result.position_x.min == 1
    assert result.position_x.max == 5

    result = range_from_collection(collection.references, loc_properties="position_x")
    assert result.position_x.min == 1
    assert result.position_x.max == 5

    result = range_from_collection(collection.references, loc_properties=True)
    assert result.position_x.min == 1
    assert result.position_x.max == 5
    assert result.frame.min == 1


def test__ranges(locdata_2d):
    ranges_ = ranges(locdata_2d)
    assert np.array_equal(ranges_, [[1, 5], [1, 6]])
    ranges_ = ranges(locdata_2d, loc_properties=True)
    assert np.array_equal(ranges_, [[1, 5], [1, 6], [1, 6], [80, 150]])
    ranges_ = ranges(locdata_2d, loc_properties="intensity")
    assert np.array_equal(ranges_, [[80, 150]])
    ranges_ = ranges(locdata_2d, loc_properties=["intensity"])
    assert np.array_equal(ranges_, [[80, 150]])
    ranges_ = ranges(locdata_2d, loc_properties=["frame", "intensity"])
    assert np.array_equal(ranges_, [[1, 6], [80, 150]])
    ranges_ = ranges(locdata_2d, loc_properties=("frame", "intensity"))
    assert np.array_equal(ranges_, [[1, 6], [80, 150]])
    ranges_ = ranges(locdata_2d, loc_properties=None, special="zero")
    assert np.array_equal(ranges_, [[0, 5], [0, 6]])
    ranges_ = ranges(locdata_2d, loc_properties=True, special="link")
    assert np.array_equal(ranges_, [[1, 150], [1, 150], [1, 150], [1, 150]])


def test__ranges_empty(locdata_empty, locdata_single_localization):
    assert ranges(locdata_empty) is None
    ranges_ = ranges(locdata_single_localization, loc_properties=None)
    assert np.array_equal(ranges_, [[1, 2], [1, 2]])
    ranges_ = ranges(locdata_single_localization, loc_properties=True)
    assert np.array_equal(ranges_, [[1, 2], [1, 2], [1, 2], [1, 2]])
