from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.io.io_locdata import load_txt_file


# fixtures for random points

@pytest.fixture(scope='session')
def few_random_points():
    points = np.array([[0.066, 0.64], [0.92, 0.65], [0.11, 0.40], [0.20, 0.17], [0.75, 0.92],
                       [0.01, 0.12], [0.23, 0.54], [0.05, 0.25], [0.70, 0.73], [0.43, 0.16]])
    return points


# fixtures for LocData objects

@pytest.fixture(scope='session')
def locdata_empty():
    df = pd.DataFrame()
    return LocData.from_dataframe(dataframe=df, meta={'creation_date': 1000000001})


@pytest.fixture(scope='session')
def locdata_single_localization():
    locdata_dict = {
        'position_x': np.array([1]),
        'position_y': np.array([1]),
        'frame': np.array([1]),
        'intensity': np.array([1]),
    }
    df = pd.DataFrame(locdata_dict)
    return LocData.from_dataframe(dataframe=df, meta={'creation_date': 1000000001})


@pytest.fixture(scope='session')
def locdata_fix():
    locdata_dict = {
        'position_x': np.array([1, 1, 2, 3, 4, 5]),
        'position_y': np.array([1, 5, 3, 6, 2, 5]),
        'frame': np.array([1, 2, 2, 4, 5, 6]),
        'intensity': np.array([100, 150, 110, 80, 105, 95]),
    }
    df = pd.DataFrame(locdata_dict)
    return LocData.from_dataframe(dataframe=df, meta={'creation_date': 1000000001})


@pytest.fixture(scope='session')
def locdata_non_standard_index():
    locdata_dict = {
        'position_x': np.array([1, 1, 2, 3, 4, 5]),
        'position_y': np.array([1, 5, 3, 6, 2, 5]),
        'frame': np.array([1, 2, 2, 4, 5, 6]),
        'intensity': np.array([100, 150, 110, 80, 105, 95]),
    }
    df = pd.DataFrame(locdata_dict)
    df.index = [2, 1, 5, 10, 100, 200]
    return LocData.from_dataframe(dataframe=df, meta={'creation_date': 1000000001})


@pytest.fixture(scope='session')
def locdata_blobs_2d():
    path = Path(ROOT_DIR / 'tests/Test_data/five_blobs.txt')
    dat = load_txt_file(path)
    dat.meta.creation_date = 1000000001
    return dat


@pytest.fixture(scope='session')
def locdata_blobs_3d():
    path = Path(ROOT_DIR / 'tests/Test_data/five_blobs_3D.txt')
    dat = load_txt_file(path, meta={'creation_date': 1000000001})
    return dat
