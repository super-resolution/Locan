import pytest
import numpy as np
import pandas as pd

from surepy import LocData

# fixtures

@pytest.fixture(scope='session')
def few_random_points():
    points = np.array([[0.066, 0.64], [0.92, 0.65], [0.11, 0.40], [0.20, 0.17], [0.75, 0.92],
                   [0.01, 0.12], [0.23, 0.54], [0.05, 0.25], [0.70, 0.73], [0.43, 0.16]])
    return points


@pytest.fixture(scope='session')
def locdata_fix():
    dict = {
        'Position_x': np.array([1,1,2,3,4,5]),
        'Position_y': np.array([1,5,3,6,2,5]),
        'Frame': np.array([1, 2, 2, 4, 5, 6]),
        'Intensity': np.array([100, 150, 110, 80, 105, 95]),
    }
    df = pd.DataFrame(dict)
    return LocData.from_dataframe(dataframe=df, meta={'creation_date': 1000000001})
