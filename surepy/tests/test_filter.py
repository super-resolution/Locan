import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.data.filter import select_by_condition


@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))

def test_select_by_condition(locdata_simple):
    dat_s = select_by_condition(locdata_simple, 'Position_x>1')
    assert (len(dat_s) == 2)