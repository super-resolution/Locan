import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.data.transformation import randomize
from surepy.data.hulls import  Bounding_box, Convex_hull_scipy


@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))

def test_randomize(locdata_simple):
    locdata_randomized = randomize(locdata_simple, hull_region='bb')
    #locdata_randomized.print_meta()
    assert (len(locdata_randomized) == len(locdata_simple))
