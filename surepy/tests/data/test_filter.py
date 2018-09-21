import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.data.rois import Roi_manager
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
    assert (len(dat_s) == 2)
    # dat_s.print_meta()

def test_random_subset(locdata_simple):
    dat_s = random_subset(locdata_simple, number_points=3)
    assert (len(dat_s) == 3)
    # dat_s.print_meta()
    # print(dat_s.data)

def test_rois (locdata_simple):
    roim = Roi_manager()
    roim.add_rectangle((0,3))
    dat_1 = select_by_region(locdata_simple, roi=roim.rois[0])
    assert(len(dat_1)==6)
    roim.clear()
    roim.add_rectangle((0,3,0,3))
    dat_1 = select_by_region(locdata_simple, roi=roim.rois[0])
    assert(len(dat_1)==5)
    roim.clear()
    roim.add_rectangle((0,3,0,3,0,3))
    dat_1 = select_by_region(locdata_simple, roi=roim.rois[0])
    assert(len(dat_1)==4)