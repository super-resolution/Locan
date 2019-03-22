import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.data.properties import statistics


@pytest.fixture()
def locdata_simple():
    dict = {
        'position_x': [0, 0, 1, 4, 5],
        'position_y': [0, 1, 3, 4, 1]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


def test_statistics(locdata_simple):
    stat = statistics(locdata=locdata_simple)
    assert stat == {'position_x_count': 5.0, 'position_x_min': 0.0, 'position_x_max': 5.0, 'position_x_mean': 2.0,
                   'position_x_median': 1.0, 'position_x_std': 2.3452078799117149,
                   'position_x_sem': 1.0488088481701516, 'position_y_count': 5.0, 'position_y_min': 0.0,
                   'position_y_max': 4.0, 'position_y_mean': 1.8, 'position_y_median': 1.0,
                   'position_y_std': 1.6431676725154984, 'position_y_sem': 0.73484692283495345}

def test_statistics_from_dataframe(locdata_simple):
    stat = statistics(locdata=locdata_simple.data)
    assert stat == {'position_x_count': 5.0, 'position_x_min': 0.0, 'position_x_max': 5.0, 'position_x_mean': 2.0,
                   'position_x_median': 1.0, 'position_x_std': 2.3452078799117149,
                   'position_x_sem': 1.0488088481701516, 'position_y_count': 5.0, 'position_y_min': 0.0,
                   'position_y_max': 4.0, 'position_y_mean': 1.8, 'position_y_median': 1.0,
                   'position_y_std': 1.6431676725154984, 'position_y_sem': 0.73484692283495345}

def test_statistics_with_one_statfunction(locdata_simple):
    stat = statistics(locdata=locdata_simple, statistic_keys=('min'))
    assert stat == {'position_x_min': 0, 'position_y_min': 0}

def test_statistics_from_Series(locdata_simple):
    stat = statistics(locdata=locdata_simple.data['position_x'], statistic_keys=('mean'))
    assert stat['position_x_mean'] == 2
    stat = statistics(locdata=locdata_simple.data['position_x'], statistic_keys=('min', 'max'))
    assert stat == {'position_x_min': 0, 'position_x_max': 5}

def test_statistics_with_new_column(locdata_simple):
    locdata_simple.dataframe = locdata_simple.dataframe.assign(new= np.arange(5))
    stat = statistics(locdata=locdata_simple)
    # print(stat)
    assert stat == {'position_x_count': 5.0, 'position_x_min': 0.0, 'position_x_max': 5.0, 'position_x_mean': 2.0,
                    'position_x_median': 1.0, 'position_x_std': 2.3452078799117149,
                    'position_x_sem': 1.0488088481701516, 'position_y_count': 5.0, 'position_y_min': 0.0,
                    'position_y_max': 4.0, 'position_y_mean': 1.8, 'position_y_median': 1.0,
                    'position_y_std': 1.6431676725154984, 'position_y_sem': 0.73484692283495345,
                    'new_count': 5.0, 'new_min': 0.0, 'new_max': 4.0, 'new_mean': 2.0, 'new_median': 2.0,
                    'new_std': 1.5811388300841898, 'new_sem': 0.70710678118654757}

