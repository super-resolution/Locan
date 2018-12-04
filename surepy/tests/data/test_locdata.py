import pytest
import numpy as np
import pandas as pd

from surepy.data import metadata_pb2
from surepy.data.locdata import LocData


# fixtures

@pytest.fixture()
def df_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1]
    }
    return pd.DataFrame.from_dict(dict)

@pytest.fixture()
def df_line():
    dict = {
        'Position_x': [1, 2, 3, 4, 5],
        'Position_y': [1, 2, 3, 4, 5]
    }
    return pd.DataFrame.from_dict(dict)

@pytest.fixture()
def df_other_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [10, 11, 13, 14, 11]
        }
    return pd.DataFrame.from_dict(dict)


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
    assert (len(dat) == 5)
    assert (dat.meta.comment == COMMENT_METADATA.comment)

def test_LocData_from_dataframe_with_meta_dict(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta={'comment':'some user comment'})
    assert (len(dat) == 5)
    assert (dat.meta.comment == COMMENT_METADATA.comment)

# todo: identify assertion failure
# def test_LocData_class_count(df_simple):
#     dat_1 = LocData.from_dataframe(dataframe=df_simple)
#     assert (LocData.count == 1)
#     dat_2 = LocData.from_dataframe(dataframe=df_simple)
#     # print(dat_2.properties)
#     assert(dat_1.properties == dat_2.properties)
#     assert (LocData.count == 2)
#     del(dat_1)
#     assert (LocData.count == 1)

def test_LocData_from_selection(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple)
    sel = LocData.from_selection(locdata=dat, indices=[1,3,4], meta=COMMENT_METADATA)
    assert (len(sel) == 3)
    assert (sel.meta.comment == COMMENT_METADATA.comment)

def test_LocData_from_collection(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple)
    sel_1 = LocData.from_selection(locdata=dat, indices=[0, 1, 2])
    sel_2 = LocData.from_selection(locdata=dat, indices=[3, 4])
    col = LocData.from_collection([sel_1, sel_2], meta=COMMENT_METADATA)
    assert (len(col.references) == 2)
    assert (len(col) == 2)
    assert (col.meta.comment == 'some user comment')

def test_LocData_concat(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple)
    sel_1 = LocData.from_selection(locdata=dat, indices=[0, 1, 2])
    sel_2 = LocData.from_selection(locdata=dat, indices=[3, 4])
    col = LocData.concat([sel_1, sel_2], meta=COMMENT_METADATA)
    assert (len(col) == 5)
    assert (col.meta.comment == 'some user comment')

def test_LocData_reduce(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    sel_1 = LocData.from_selection(locdata=dat, indices=[1,3,4], meta=COMMENT_METADATA)
    sel_2 = LocData.from_selection(locdata=dat, indices=[1,3,4], meta=COMMENT_METADATA)
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
    sel.dataframe = sel.dataframe.assign(new= np.arange(3))
    assert all(list(sel.data.columns == ['index', 'Position_x', 'Position_y', 'new']))

    # from collection
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    sel_1 = LocData.from_selection(locdata=dat, indices=[0, 1, 2])
    sel_2 = LocData.from_selection(locdata=dat, indices=[3, 4])
    col = LocData.from_collection([sel_1, sel_2], meta=COMMENT_METADATA)
    col.dataframe = col.dataframe.assign(new= np.arange(2))
    # print(col.data.columns)
    assert all(list(col.data.columns == ['Localization_count', 'Localization_density_bb', 'Position_x', 'Position_y', 'Region_measure_bb', 'Subregion_measure_bb', 'new']))


# locdata and metadata

def test_LocData_handling_metadata(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    dat.meta.comment = 'new comment'
    assert (dat.meta.comment == 'new comment')

    dat.meta.map['variable key'] = 'new comment'
    # print(dat.meta.map)
    assert (dat.meta.map == {'variable key': 'new comment'})



