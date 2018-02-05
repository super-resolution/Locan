import pytest
import numpy as np
import pandas as pd
# from surepy.data.locdata import LocData

# fixtures

@pytest.fixture(scope='session')
def few_random_points():
    points = np.array([[0.066, 0.64], [0.92, 0.65], [0.11, 0.40], [0.20, 0.17], [0.75, 0.92],
                   [0.01, 0.12], [0.23, 0.54], [0.05, 0.25], [0.70, 0.73], [0.43, 0.16]])
    return points