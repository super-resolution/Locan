import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.analysis import Nearest_neighbor_distances

# fixtures

@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))

@pytest.fixture()
def other_locdata_simple():
    dict = {
        'Position_x': [10, 11],
        'Position_y': [10, 11],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


# tests

def test_Nearest_neighbor_distances(locdata_simple, other_locdata_simple):
    nn_1 = Nearest_neighbor_distances(locdata_simple)
    # print(nn_1.results)
    assert(nn_1.results['nn_index'].iloc[0] == 1)

    nn_2 = Nearest_neighbor_distances(locdata_simple, other_locdata=other_locdata_simple)
    # print(nn_2.results)
    assert(nn_2.results['nn_distance'].iloc[0] == 14.142135623730951)
