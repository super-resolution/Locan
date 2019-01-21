import pytest
import pandas as pd
from surepy import LocData
from surepy.data.rois import Roi
from surepy.data.filter import select_by_condition, random_subset, select_by_region, exclude_sparse_points
from surepy.data.transform import transform_affine


@pytest.fixture()
def locdata_simple():
    dict_ = {
        'Position_x': [0, 1, 2, 3, 0, 1, 4, 5],
        'Position_y': [0, 1, 2, 3, 1, 4, 5, 1],
        'Position_z': [0, 1, 2, 3, 4, 4, 4, 5]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict_))


def test_select_by_condition(locdata_simple):
    dat_s = select_by_condition(locdata_simple, 'Position_x>1')
    assert (len(dat_s) == 4)
    # dat_s.print_meta()
    # print(dat_s.meta)


def test_LocData_selection_from_collection(locdata_simple):
    # print(locdata_simple.meta)
    sel = []
    for i in range(4):
        sel.append(select_by_condition(locdata_simple, f'Position_x>{i}'))
    col = LocData.from_collection(sel)
    assert (len(col) == 4)
    assert (len(col.references) == 4)
    # print(col.references[0].meta)
    # print(col.data)
    # print(col.meta)

    col_sel = select_by_condition(col, 'Localization_count>2')
    assert (len(col_sel) == 3)
    # print(col_sel.data)
    assert(col_sel.references is col)

    col_sel_sel = select_by_condition(col_sel, 'Localization_count<4')
    assert (len(col_sel_sel) == 1)
    # print(col_sel_sel.data)
    assert(col_sel_sel.references is col_sel)
    # print(col_sel_sel.meta)


def test_random_subset(locdata_simple):
    dat_s = random_subset(locdata_simple, number_points=3)
    assert (len(dat_s) == 3)
    # dat_s.print_meta()
    # print(dat_s.data)
    # print(dat_s.meta)


def test_select_by_region(locdata_simple):
    roi_dict = dict(points=(0, 3), type='rectangle')
    dat_1 = select_by_region(locdata_simple, roi=roi_dict)
    assert(len(dat_1) == 6)
    roi_dict = dict(points=(0, 3, 0, 3), type='rectangle')
    dat_1 = select_by_region(locdata_simple, roi=roi_dict)
    assert(len(dat_1) == 5)
    roi_dict = dict(points=(0, 3, 0, 3, 0, 3), type='rectangle')
    dat_1 = select_by_region(locdata_simple, roi=roi_dict)
    assert(len(dat_1) == 4)

    roi = Roi(points=(0, 3), type='rectangle')
    dat_1 = select_by_region(locdata_simple, roi=roi)
    assert(len(dat_1) == 6)
    roi = dict(points=(0, 3, 0, 3), type='rectangle')
    dat_1 = select_by_region(locdata_simple, roi=roi)
    assert(len(dat_1) == 5)
    roi = dict(points=(0, 3, 0, 3, 0, 3), type='rectangle')
    dat_1 = select_by_region(locdata_simple, roi=roi)
    assert(len(dat_1) == 4)


def test_exclude_sparse_points(locdata_simple):
    new_locdata = exclude_sparse_points(locdata=locdata_simple, radius=3, min_samples=2)
    assert len(new_locdata) == 4
    locdata_simple_trans = transform_affine(locdata_simple, offset=(0.5, 0, 0))
    new_locdata = exclude_sparse_points(locdata=locdata_simple,
                                        other_locdata=locdata_simple_trans, radius=2, min_samples=2)
    assert len(new_locdata) == 3
    # print(new_locdata.meta)
