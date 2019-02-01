import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.data.properties import max_distance


@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 0, 0],
        'Position_y': [0, 1, 3]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


def test_max_distance(locdata_simple):
    mdist = max_distance(locdata=locdata_simple)
    assert (mdist == {'Max_distance': 3})
