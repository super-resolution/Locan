import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.data.rois import Roi
from surepy.simulation import simulate_blobs
from surepy.data.filter import select_by_condition, random_subset, select_by_region


@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 1, 2, 3, 0, 1, 4, 5],
        'Position_y': [0, 1, 2, 3, 1, 4, 5, 1],
        'Position_z': [0, 1, 2, 3, 4, 4, 4, 5]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))

def test_select_by_condition(locdata_simple):
    dat_s = select_by_condition(locdata_simple, 'Position_x>1')
    assert (len(dat_s) == 4)
    # dat_s.print_meta()

def test_random_subset(locdata_simple):
    dat_s = random_subset(locdata_simple, number_points=3)
    assert (len(dat_s) == 3)
    # dat_s.print_meta()
    # print(dat_s.data)

def test_select_by_region (locdata_simple):
    roi_dict = dict(points=(0,3), type='rectangle')
    dat_1 = select_by_region(locdata_simple, roi=roi_dict)
    assert(len(dat_1)==6)
    roi_dict = dict(points=(0,3,0,3), type='rectangle')
    dat_1 = select_by_region(locdata_simple, roi=roi_dict)
    assert(len(dat_1)==5)
    roi_dict = dict(points=(0,3,0,3,0,3), type='rectangle')
    dat_1 = select_by_region(locdata_simple, roi=roi_dict)
    assert(len(dat_1)==4)

    roi = Roi(points=(0,3), type='rectangle')
    dat_1 = select_by_region(locdata_simple, roi=roi)
    assert(len(dat_1)==6)
    roi = dict(points=(0,3,0,3), type='rectangle')
    dat_1 = select_by_region(locdata_simple, roi=roi)
    assert(len(dat_1)==5)
    roi = dict(points=(0,3,0,3,0,3), type='rectangle')
    dat_1 = select_by_region(locdata_simple, roi=roi)
    assert(len(dat_1)==4)