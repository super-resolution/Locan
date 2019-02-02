import pytest
import numpy as np
import pandas as pd

from surepy.data import metadata_pb2
from surepy.data.locdata import LocData
from surepy.data.rois import RoiRegion


# fixtures

@pytest.fixture()
def df_simple():
    dict_ = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1]
    }
    return pd.DataFrame.from_dict(dict_)


@pytest.fixture()
def df_line():
    dict_ = {
        'Position_x': [1, 2, 3, 4, 5],
        'Position_y': [1, 2, 3, 4, 5]
    }
    return pd.DataFrame.from_dict(dict_)


@pytest.fixture()
def df_other_simple():
    dict_ = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [10, 11, 13, 14, 11]
        }
    return pd.DataFrame.from_dict(dict_)


# tests

COMMENT_METADATA = metadata_pb2.Metadata(comment='some user comment')


def test_LocData(df_simple):
    dat = LocData(dataframe=df_simple, meta=COMMENT_METADATA)
    assert (len(dat) == 5)
    assert (dat.coordinate_labels == ['Position_x', 'Position_y'])
    for x, y in zip(dat.coordinates, [[0, 0], [0, 1], [1, 3], [4, 4], [5, 1]]):
        assert np.all(x == np.array(y))
    assert (dat.meta.comment == COMMENT_METADATA.comment)
    # dat.print_meta()
    # dat.print_summary()


def test_LocData_from_dataframe(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    assert(list(dat.properties.keys()) == ['Localization_count', 'Position_x', 'Position_y',
                                           'Region_measure_bb', 'Subregion_measure_bb', 'Localization_density_bb'])
    assert (len(dat) == 5)
    assert (dat.meta.comment == COMMENT_METADATA.comment)


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

    sel_sel = LocData.from_selection(locdata=sel, indices=[0, 2], meta={'comment': 'Selection of a selection.'})
    print(sel_sel.meta.comment)
    assert (len(sel_sel) == 2)
    assert (sel_sel.references is sel)
    print(sel_sel.data)


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

    col_sel_sel = LocData.from_selection(locdata=col_sel, indices=[1])
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
    dat.dataframe = dat.dataframe.assign(new= np.arange(5))
    assert(dat.data.equals(dat.dataframe))

    # from selection
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    sel = LocData.from_selection(locdata=dat, indices=[0, 1, 2])
    sel.dataframe = sel.dataframe.assign(new=np.arange(3))
    assert(list(sel.dataframe.columns) == ['new'])
    assert(list(sel.references.dataframe.columns) == ['Position_x', 'Position_y'])
    assert all(list(sel.data.columns == ['Position_x', 'Position_y', 'new']))
    sel.reduce()
    assert all(list(sel.dataframe.columns == ['Position_x', 'Position_y', 'new']))

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
    assert all(list(col.data.columns == ['Localization_count', 'Localization_density_bb', 'Position_x', 'Position_y',
                                         'Region_measure_bb', 'Subregion_measure_bb', 'new']))


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
    assert dat.region.region_measure==2
    print(dat.properties)
