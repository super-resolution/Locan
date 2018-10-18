import pytest

import numpy as np
import pandas as pd

from surepy import LocData
from surepy.analysis.pipeline import Pipeline, Pipeline_test

# fixtures

@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


# tests

def test_Pipeline_test(locdata_simple):
    pipe = Pipeline(locdata_simple)
    assert(pipe.file is None)

    pipe = Pipeline_test(locdata_simple)
    assert(pipe.test is None)
    pipe.compute()
    assert(pipe.test is True)

