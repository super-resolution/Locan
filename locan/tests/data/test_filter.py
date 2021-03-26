import pytest
import numpy as np
import pandas as pd
from locan import LocData
from locan.data.rois import RoiRegion
from locan.data.filter import select_by_condition, random_subset, select_by_region, exclude_sparse_points
from locan.data.transform import transform_affine


@pytest.fixture()
def locdata_simple_():
    locdata_dict = {
        'position_x': [0, 1, 2, 3, 0, 1, 4, 5],
        'position_y': [0, 1, 2, 3, 1, 4, 5, 1],
        'position_z': [0, 1, 2, 3, 4, 4, 4, 5]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


@pytest.fixture()
def locdata_simple():
    locdata_dict = {
        'position_x': [0, 1, 2, 3, 0, 1, 4, 5],
        'position_y': [0, 1, 2, 3, 1, 4, 5, 1],
        'position_z': [0, 1, 2, 3, 4, 4, 4, 5]
    }
    df = pd.DataFrame(locdata_dict)
    df.index = [2, 0, 1, 3, 4, 5, 6, 7]
    return LocData.from_dataframe(dataframe=df, meta={'creation_date': "1111-11-11 11:11:11 +0100"})


def test_select_by_condition(locdata_simple):
    dat_s = select_by_condition(locdata_simple, 'position_x>1')
    assert (len(dat_s) == 4)
    assert np.all(dat_s.data.index == [1, 3, 6, 7])
    # dat_s.print_meta()
    # print(dat_s.meta)
    # print(dat_s.data)


def test_LocData_selection_from_collection(locdata_simple):
    # print(locdata_simple.meta)
    sel = []
    for i in range(4):
        sel.append(select_by_condition(locdata_simple, f'position_x>{i}'))
    col = LocData.from_collection(sel)
    assert (len(col) == 4)
    assert (len(col.references) == 4)
    # print(col.references[0].meta)
    # print(col.data)
    # print(col.meta)

    col_sel = select_by_condition(col, 'localization_count>2')
    assert (len(col_sel) == 3)
    # print(col_sel.data)
    assert(col_sel.references is col)

    col_sel_sel = select_by_condition(col_sel, 'localization_count<4')
    assert (len(col_sel_sel) == 1)
    # print(col_sel_sel.data)
    assert(col_sel_sel.references is col_sel)
    # print(col_sel_sel.meta)


def test_random_subset(locdata_simple):
    dat_s = random_subset(locdata_simple, n_points=3)
    assert (len(dat_s) == 3)
    # dat_s.print_meta()
    # print(dat_s.data)
    # print(dat_s.meta)


def test_select_by_region(locdata_simple):
    roi_dict = dict(region_type='rectangle', region_specs=((0, 0), 2, 1, 10))
    dat = select_by_region(locdata=locdata_simple, region=roi_dict, properties_for_roi=['position_y', 'position_z'])
    assert(len(dat) == 1)

    roi_region = RoiRegion(region_type='rectangle', region_specs=((0, 0), 2, 1, 10))
    dat = select_by_region(locdata_simple, region=roi_region)
    assert(len(dat) == 2)


def test_exclude_sparse_points(locdata_simple):
    new_locdata = exclude_sparse_points(locdata=locdata_simple, radius=3, min_samples=2)
    assert len(new_locdata) == 4
    # todo: check if the correct points are taken
    locdata_simple_trans = transform_affine(locdata_simple, offset=(0.5, 0, 0))
    new_locdata = exclude_sparse_points(locdata=locdata_simple,
                                        other_locdata=locdata_simple_trans, radius=2, min_samples=2)
    assert len(new_locdata) == 3
    # print(new_locdata.meta)
