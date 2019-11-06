import pytest
import numpy as np
import pandas as pd

from surepy.data import metadata_pb2
from surepy.data.locdata import LocData
from surepy.data.rois import RoiRegion

# todo check for selections that return zero data points
# todo check for empty dataframe
# todo check for single localization


# fixtures

@pytest.fixture()
def df_simple():
    dict_ = {
        'position_x': [0, 0, 1, 4, 5],
        'position_y': [0, 1, 3, 4, 1]
    }
    return pd.DataFrame.from_dict(dict_)


@pytest.fixture()
def df_line():
    dict_ = {
        'position_x': [1, 2, 3, 4, 5],
        'position_y': [1, 2, 3, 4, 5]
    }
    return pd.DataFrame.from_dict(dict_)


@pytest.fixture()
def df_empty():
    dict_ = {
    }
    return pd.DataFrame.from_dict(dict_)


@pytest.fixture()
def df_other_simple():
    dict_ = {
        'position_x': [0, 0, 1, 4, 5],
        'position_y': [10, 11, 13, 14, 11]
        }
    return pd.DataFrame.from_dict(dict_)


# tests

COMMENT_METADATA = metadata_pb2.Metadata(comment='some user comment')


def test_LocData(df_simple):
    dat = LocData(dataframe=df_simple, meta=COMMENT_METADATA)
    assert len(dat) == 5
    assert dat.coordinate_labels == ['position_x', 'position_y']
    assert dat.dimensions == 2
    assert np.array_equal(dat.centroid, [2., 1.8])
    for x, y in zip(dat.coordinates, [[0, 0], [0, 1], [1, 3], [4, 4], [5, 1]]):
        assert np.all(x == np.array(y))
    assert dat.meta.comment == COMMENT_METADATA.comment
    # assert dat.meta.identifier == '1'  # this test runs ok for this testing this function alone.


def test_LocData_empty(df_empty):
    dat = LocData(dataframe=df_empty)
    assert len(dat) == 0
    assert dat.coordinate_labels == []
    assert dat.dimensions == 0


def test_LocData_from_dataframe(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    # print(dat.properties.keys())
    assert list(dat.properties.keys()) == ['localization_count', 'position_x', 'position_y', 'region_measure_bb',
                                           'localization_density_bb', 'subregion_measure_bb']
    assert len(dat) == 5
    assert dat.meta.comment == COMMENT_METADATA.comment


def test_LocData_from_dataframe_empty(df_empty):
    dat = LocData.from_dataframe(dataframe=df_empty)
    assert len(dat) == 0
    assert dat.coordinate_labels == []
    # print(dat.data)


# this test is not running wihtin complete test run. But it works when run by itself.
# def test_LocData_count(df_simple):
#     dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
#     assert (LocData.count == 1)
#     dat_2 = LocData.from_dataframe(dataframe=df_simple)
#     assert(dat.properties == dat_2.properties)
#     assert (LocData.count == 2)
#     del(dat)
#     assert (LocData.count == 1)


def test_LocData_from_dataframe_with_meta_dict(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta={'comment': 'some user comment'})
    assert (len(dat) == 5)
    assert (dat.references is None)
    assert (dat.meta.comment == COMMENT_METADATA.comment)


def test_LocData_from_selection(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple)
    sel = LocData.from_selection(locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA)
    assert (len(sel) == 3)
    assert (sel.references is dat)
    assert (sel.meta.comment == COMMENT_METADATA.comment)

    sel_sel = LocData.from_selection(locdata=sel, indices=[1, 3], meta={'comment': 'Selection of a selection.'})
    assert sel_sel.meta.comment == 'Selection of a selection.'
    assert (len(sel_sel) == 2)
    assert (sel_sel.references is sel)

    sel_empty = LocData.from_selection(locdata=dat, indices=[], meta={'comment': 'This selection is empty.'})
    assert sel_empty.data.index.values.size == 0


def test_LocData_from_collection(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple)
    sel_1 = LocData.from_selection(locdata=dat, indices=[0, 1, 2])
    sel_2 = LocData.from_selection(locdata=dat, indices=[3, 4])
    col = LocData.from_collection([sel_1, sel_2], meta=COMMENT_METADATA)
    assert (len(col.references) == 2)
    assert (len(col) == 2)
    assert (col.meta.comment == COMMENT_METADATA.comment)
    # print(col.properties)


def test_LocData_selection_from_collection(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple)
    sel = []
    col = LocData.from_collection(sel)
    assert len(col) == 0

    for i in range(4):
        sel.append(LocData.from_selection(locdata=dat, indices=[i]))
    col = LocData.from_collection(sel, meta=COMMENT_METADATA)
    assert (len(col) == 4)
    assert (len(col.references) == 4)
    # print(col.data)

    col_sel = LocData.from_selection(locdata=col, indices=[0, 2, 3])
    assert (len(col_sel) == 3)
    assert(col_sel.references is col)
    # print(col_sel.data)

    col_sel_sel = LocData.from_selection(locdata=col_sel, indices=[2])
    assert (len(col_sel_sel) == 1)
    assert(col_sel_sel.references is col_sel)
    # print(col_sel_sel.data)


def test_LocData_concat(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple)
    sel_1 = LocData.from_selection(locdata=dat, indices=[0, 1, 2])
    sel_2 = LocData.from_selection(locdata=dat, indices=[3, 4])
    col = LocData.concat([sel_1, sel_2], meta=COMMENT_METADATA)
    assert (len(col) == 5)
    assert (col.meta.comment == COMMENT_METADATA.comment)
    assert len(col.references) == 2
    sel_1.reduce()
    col = LocData.concat([sel_1, sel_2], meta=COMMENT_METADATA)
    assert len(col.references) == 2
    sel_1.reduce()
    sel_2.reduce()
    col = LocData.concat([sel_1, sel_2], meta=COMMENT_METADATA)
    assert col.references is None


def test_LocData_reduce(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    sel_1 = LocData.from_selection(locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA)
    sel_2 = LocData.from_selection(locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA)
    flag = sel_1.reduce()
    assert (flag == 1)
    assert (len(sel_1) == 3)
    assert (len(sel_1.data) == len(sel_2.data))


# locdata with added columns

def test_LocData_add_column_to_dataframe(df_simple):
    # from dataframe
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    dat.dataframe = dat.dataframe.assign(new=np.arange(5))
    assert(dat.data.equals(dat.dataframe))

    # from selection
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    sel = LocData.from_selection(locdata=dat, indices=[0, 1, 2])
    sel.dataframe = sel.dataframe.assign(new=np.arange(3))
    assert(list(sel.dataframe.columns) == ['new'])
    assert(list(sel.references.dataframe.columns) == ['position_x', 'position_y'])
    assert all(list(sel.data.columns == ['position_x', 'position_y', 'new']))
    sel.reduce()
    assert all(list(sel.dataframe.columns == ['position_x', 'position_y', 'new']))

    # sel_2 = LocData.from_selection(locdata=sel, indices=[0, 2])
    # print(sel_2.data)
    # print(sel_2.references.data)
    # print(sel_2.references.references.data)
    # sel_2.reduce()
    # print(sel_2.data)

    # from collection
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    sel_1 = LocData.from_selection(locdata=dat, indices=[0, 1, 2])
    sel_2 = LocData.from_selection(locdata=dat, indices=[3, 4])
    col = LocData.from_collection([sel_1, sel_2], meta=COMMENT_METADATA)
    col.dataframe = col.dataframe.assign(new=np.arange(2))
    # print(col.data.columns)
    assert all(list(col.data.columns == ['localization_count', 'position_x', 'position_y', 'region_measure_bb',
                                         'localization_density_bb', 'subregion_measure_bb', 'new']))


# locdata and metadata

def test_LocData_handling_metadata(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    dat.meta.comment = 'new comment'
    assert (dat.meta.comment == 'new comment')

    dat.meta.map['variable key'] = 'new comment'
    # print(dat.meta.map)
    assert (dat.meta.map == {'variable key': 'new comment'})


# locdata and regions

def test_locdata_region(df_simple):
    roi_dict = dict(region_type='rectangle', region_specs=((0, 0), 2, 1, 10))
    roi_region = RoiRegion(**roi_dict)
    dat = LocData.from_dataframe(dataframe=df_simple)
    dat.region = roi_region
    assert isinstance(dat._region, RoiRegion)
    dat.region = roi_dict
    assert isinstance(dat._region, RoiRegion)
    assert dat.region.region_measure == 2


# standard LocData fixtures

@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', 0),
    ('locdata_single_localization', 1),
    ('locdata_fix', 6),
    ('locdata_non_standard_index', 6)
])
def test_standard_locdata_objects(
        locdata_empty, locdata_single_localization, locdata_fix, locdata_non_standard_index,
        fixture_name, expected):
    dat = eval(fixture_name)
    assert len(dat) == expected


@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', pytest.raises(KeyError)),
    ('locdata_single_localization', pytest.raises(KeyError)),
])
def test_locdata_from_selection_exceptions(
        locdata_empty, locdata_single_localization, locdata_fix, locdata_non_standard_index,
        fixture_name, expected):
    dat = eval(fixture_name)
    with expected:
        LocData.from_selection(locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA)


@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_fix', 3),
    ('locdata_non_standard_index', 3)
])
def test_locdata_from_selection_(
        locdata_empty, locdata_single_localization, locdata_fix, locdata_non_standard_index,
        fixture_name, expected):
    dat = eval(fixture_name)
    sel = LocData.from_selection(locdata=dat, indices=[1, 2, 5], meta=COMMENT_METADATA)
    assert (len(sel) == expected)
    assert list(sel.data.index) == [1, 2, 5]
    assert (sel.references is dat)
    assert (sel.meta.comment == COMMENT_METADATA.comment)
