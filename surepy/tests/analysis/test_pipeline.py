import pytest

import numpy as np
import pandas as pd

from surepy import LocData
from surepy.analysis.pipeline import Pipeline, compute_test

# fixtures

@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


# tests

def test_Pipeline(locdata_simple):
    ''' use Pipeline by passing an instance '''
    pipe = Pipeline(locdata_simple)
    compute_test(pipe)
    assert(pipe.test==True)
    # print(f'results: {pipe.test}')
    # print(f'meta: {pipe.meta}')

def test_Pipeline_2(locdata_simple):
    ''' use Pipeline by inheritance - recommended. '''
    class MyPipe(Pipeline):
        compute = compute_test

    pipe = MyPipe(locdata_simple)
    pipe.compute()
    assert(pipe.test==True)
    # print(f'results: {pipe.test}')
    # print(f'meta: {pipe.meta}')
