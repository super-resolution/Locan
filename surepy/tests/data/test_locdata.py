import copy

import pytest
import numpy as np
import pandas as pd

from surepy.data import metadata_pb2
from surepy import LocData
from surepy import RoiRegion
from surepy import AlphaShape


# fixtures for DataFrames (fixtures for LocData are defined in conftest.py)

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
    assert dat.dimension == 2
    assert np.array_equal(dat.centroid, [2., 1.8])
    assert dat.centroid[0] == dat.properties['position_x']
    for x, y in zip(dat.coordinates, [[0, 0], [0, 1], [1, 3], [4, 4], [5, 1]]):
        assert np.all(x == np.array(y))
    assert dat.meta.comment == COMMENT_METADATA.comment
    # assert dat.meta.identifier == '1'  # the test runs ok when testing this function alone.
    assert dat.bounding_box.region_measure == 20
    assert 'region_measure_bb' in dat.properties
    assert 'localization_density_bb' in dat.properties
    assert dat.convex_hull.region_measure == 12.5
    assert 'region_measure_ch' in dat.properties
    assert 'localization_density_ch' in dat.properties
    assert round(dat.oriented_bounding_box.region_measure) == 16
    assert 'region_measure_obb' in dat.properties
    assert 'localization_density_obb' in dat.properties
    assert dat.region is None
    dat.region = dat.bounding_box.region
    assert dat.region.region_type == dat.bounding_box.region.region_type
    assert isinstance(dat.update_alpha_shape(alpha=1).alpha_shape, AlphaShape)
    assert 'region_measure_as' in dat.properties
    assert 'localization_density_as' in dat.properties


def test_LocData_empty(df_empty):
    dat = LocData(dataframe=df_empty)
    assert len(dat) == 0
    assert dat.coordinate_labels == []
    assert dat.dimension == 0
    # hulls are tested with locdata fixtures further down.


def test_LocData_from_dataframe(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    # print(dat.properties.keys())
    assert dat.bounding_box is not None
    assert list(dat.properties.keys()) == ['localization_count', 'position_x', 'position_y', 'region_measure_bb',
                                           'localization_density_bb', 'subregion_measure_bb']
    assert len(dat) == 5
    assert dat.meta.comment == COMMENT_METADATA.comment


def test_LocData_from_dataframe_empty(df_empty):
    dat = LocData.from_dataframe(dataframe=df_empty)
    assert len(dat) == 0
    assert dat.coordinate_labels == []
    # print(dat.data)


def test_LocData_count(df_simple):
    # The following is commented out because it requires time and is not needed to ensure correct functionality.
    # import gc
    # print("Current number of LocData instances: ",
    #       len([item for item in gc.get_referrers(LocData) if isinstance(item, LocData)]))
    n_instances = LocData.count

    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    assert LocData.count == 1 + n_instances
    dat_2 = LocData.from_dataframe(dataframe=df_simple)
    assert(dat.properties == dat_2.properties)
    assert LocData.count == 2 + n_instances
    del dat
    assert LocData.count == 1 + n_instances


def test_LocData_from_dataframe_with_meta_dict(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta={'comment': 'some user comment'})
    assert (len(dat) == 5)
    assert (dat.references is None)
    assert (dat.meta.comment == COMMENT_METADATA.comment)


def test_LocData_from_coordinates():
    coordinates =[(200, 500), (200, 600), (900, 650), (1000, 600)]
    dat = LocData.from_coordinates(coordinates=coordinates, meta=COMMENT_METADATA)
    assert np.array_equal(dat.coordinates, np.asarray(coordinates))
    assert all(item in ['position_x', 'position_y'] for item in dat.coordinate_labels)
    assert len(dat) == 4
    assert dat.meta.comment == COMMENT_METADATA.comment


def test_LocData_from_selection(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple)

    sel = LocData.from_selection(locdata=dat, indices=slice(1, 3))
    assert all(sel.data.index == [1, 2, 3])

    sel = LocData.from_selection(locdata=dat)
    assert len(sel) == len(df_simple)

    sel = LocData.from_selection(locdata=dat, indices=slice(4, 10))
    assert all(sel.data.index == [4])

    sel = LocData.from_selection(locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA)
    assert (len(sel) == 3)
    assert (sel.references is dat)
    assert (sel.meta.comment == COMMENT_METADATA.comment)
    assert dat.bounding_box.region_measure != sel.bounding_box.region_measure
    with pytest.raises(AttributeError):
        sel.data = df_simple

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
    assert set(col.data.columns) == {'localization_count', 'localization_density_bb', 'position_x', 'position_y',
                                     'region_measure_bb', 'subregion_measure_bb'}

    with pytest.warns(UserWarning):
        col.update_convex_hulls_in_references()
    assert set(col.data.columns) == {'localization_count', 'localization_density_bb', 'position_x', 'position_y',
                                     'region_measure_bb', 'subregion_measure_bb',
                                     'region_measure_ch', 'localization_density_ch'}
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
    sel_2.reduce()
    col = LocData.concat([sel_1, sel_2], meta=COMMENT_METADATA)
    assert col.references is None


def test_LocData_reduce(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    sel_1 = LocData.from_selection(locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA)
    sel_2 = LocData.from_selection(locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA)
    sel_1.reduce()
    assert sel_1.references is None
    assert (len(sel_1) == 3)
    assert (len(sel_1.data) == len(sel_2.data))
    col = LocData.from_collection([sel_1, sel_2], meta=COMMENT_METADATA)
    col.reduce()
    assert (len(col) == 2)
    assert col.references is None


def test_LocData_reset(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    assert dat.properties['position_x'] == 2
    assert dat.properties['region_measure_bb'] == 20
    assert (dat.meta.comment == COMMENT_METADATA.comment)

    dat.dataframe['position_x'] = [0, 1, 4, 0, 0]
    dat.dataframe['position_y'] = [1, 3, 4, 0, 0]
    dat.reset()
    assert dat.properties['position_x'] == 1
    assert dat.properties['region_measure_bb'] == 16
    assert (dat.meta.comment == COMMENT_METADATA.comment)


def test_LocData_update(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    assert dat.properties['position_x'] == 2
    assert dat.meta.element_count == 5
    new_dataframe = pd.DataFrame.from_dict(
        {'position_x': [10, 0, 1, 4],
         'position_y': [10, 1, 3, 4]}
    )
    dat.update(new_dataframe)
    pd.testing.assert_frame_equal(dat.data, new_dataframe)
    assert dat.properties['position_x'] == 3.75
    assert dat.meta.element_count == 4
    assert dat.meta.history[-1].name == "LocData.update"

    sel = LocData.from_selection(locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA)
    with pytest.warns(UserWarning):
        sel.update(new_dataframe, reset_index=True)
    pd.testing.assert_frame_equal(sel.data, new_dataframe)
    assert sel.properties['position_x'] == 3.75
    assert sel.meta.element_count == 4
    assert sel.meta.history[-1].name == "LocData.update"

    sel = LocData.from_selection(locdata=dat, indices=[1, 2, 3, 4], meta=COMMENT_METADATA)
    with pytest.warns(UserWarning):
        sel.update(new_dataframe, reset_index=True)
    pd.testing.assert_frame_equal(sel.data, new_dataframe)
    assert sel.properties['position_x'] == 3.75
    assert sel.meta.element_count == 4
    assert sel.meta.history[-1].name == "LocData.update"


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
    ('locdata_2d', 6),
    ('locdata_non_standard_index', 6)
])
def test_standard_locdata_objects(
        locdata_empty, locdata_single_localization, locdata_2d, locdata_non_standard_index,
        fixture_name, expected):
    dat = eval(fixture_name)
    assert len(dat) == expected


@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', 0),
    ('locdata_single_localization', 0),
])
def test_locdata_hulls(
        locdata_empty, locdata_single_localization,
        fixture_name, expected):
    dat = eval(fixture_name)
    assert dat.bounding_box.region_measure == 0
    with pytest.warns(UserWarning):
        assert dat.convex_hull is None
    with pytest.raises(AttributeError):
        assert dat.region_measure_ch
    if len(dat) == 0:
        assert isinstance(dat.update_alpha_shape(alpha=1).alpha_shape, AlphaShape)
    else:
        with pytest.warns(UserWarning):
            assert isinstance(dat.update_alpha_shape(alpha=1).alpha_shape, AlphaShape)


@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', 0),
    ('locdata_single_localization', 0),
])
def test_locdata_from_selection_exceptions(
        locdata_empty, locdata_single_localization, locdata_2d, locdata_non_standard_index,
        fixture_name, expected):
    dat = eval(fixture_name)
    new_dat = LocData.from_selection(locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA)
    assert len(new_dat) == expected


@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_2d', 3),
    ('locdata_non_standard_index', 3)
])
def test_locdata_from_selection_(
        locdata_empty, locdata_single_localization, locdata_2d, locdata_non_standard_index,
        fixture_name, expected):
    dat = eval(fixture_name)
    sel = LocData.from_selection(locdata=dat, indices=[1, 2, 5], meta=COMMENT_METADATA)
    assert (len(sel) == expected)
    assert set(sel.data.index) == set([1, 2, 5])
    # in coming versions: assert list(sel.data.index) == [2, 1, 5]
    assert (sel.references is dat)
    assert (sel.meta.comment == COMMENT_METADATA.comment)


def test_locdata_from_chunks_(locdata_non_standard_index):
    chunk_collection = LocData.from_chunks(locdata=locdata_non_standard_index, chunk_size=2)
    assert all(chunk_collection.references[0].data.index == [2, 1])

    chunk_collection = LocData.from_chunks(locdata=locdata_non_standard_index, n_chunks=3)
    assert all(chunk_collection.references[0].data.index == [2, 1])

    chunk_collection = LocData.from_chunks(locdata=locdata_non_standard_index, n_chunks=3, order='alternating')
    assert all(chunk_collection.references[0].data.index == [2, 10])


@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', (0, (0,))),
    ('locdata_single_localization', (1, (1,))),
    ('locdata_2d', (3, (2, 2, 2))),
    ('locdata_non_standard_index', (3, (2, 2, 2)))
])
def test_locdata_from_chunks(
        locdata_empty, locdata_single_localization, locdata_2d, locdata_non_standard_index,
        fixture_name, expected):
    dat = eval(fixture_name)
    chunk_collection = LocData.from_chunks(locdata=dat, chunk_size=2, meta=COMMENT_METADATA)
    assert isinstance(chunk_collection.references, list)
    assert len(chunk_collection) == expected[0]
    if len(chunk_collection) != 0:
        assert all(chunk_collection.data.localization_count == expected[1])
    assert chunk_collection.meta.comment == COMMENT_METADATA.comment

    chunk_collection = LocData.from_chunks(locdata=dat, chunk_size=2, order='alternating', meta=COMMENT_METADATA)
    assert isinstance(chunk_collection.references, list)
    assert len(chunk_collection) == expected[0]
    if len(chunk_collection) != 0:
        assert all(chunk_collection.data.localization_count == expected[1])
    assert chunk_collection.meta.comment == COMMENT_METADATA.comment


def test_copy():
    locdata = LocData.from_dataframe(dataframe=pd.DataFrame({'col_1': [1, 2, 3], 'col_2': ['a', 'b', 'c']}))
    selection = LocData.from_selection(locdata, indices=[0, 1])
    new_locdata = copy.copy(selection)

    assert new_locdata is not selection
    assert len(selection) == len(new_locdata)
    for attr in ['dataframe', 'indices', 'references']:
        assert getattr(selection, attr) is getattr(new_locdata, attr)
    for attr in ['properties', 'data', 'meta']:
        assert getattr(selection, attr) is not getattr(new_locdata, attr)
    pd.testing.assert_frame_equal(new_locdata.dataframe, selection.dataframe)
    pd.testing.assert_frame_equal(new_locdata.data, selection.data)
    assert new_locdata.properties == selection.properties
    assert new_locdata.indices == selection.indices
    assert new_locdata.references == selection.references
    assert new_locdata.meta != selection.meta
    assert int(new_locdata.references.meta.identifier) == int(selection.meta.identifier) - 1
    assert int(new_locdata.meta.identifier) == int(selection.meta.identifier) + 1
    assert new_locdata.meta.creation_date == selection.meta.creation_date


def test_deepcopy():
    locdata = LocData.from_dataframe(dataframe=pd.DataFrame({'col_1': [1, 2, 3], 'col_2': ['a', 'b', 'c']}))
    selection = LocData.from_selection(locdata, indices=[0, 1])
    new_locdata = copy.deepcopy(selection)

    assert new_locdata is not selection
    assert len(selection) == len(new_locdata)
    for attr in ['data', 'dataframe', 'indices', 'meta', 'properties', 'references']:
        assert getattr(selection, attr) is not getattr(new_locdata, attr)
    pd.testing.assert_frame_equal(new_locdata.dataframe, selection.dataframe)
    pd.testing.assert_frame_equal(new_locdata.data, selection.data)
    assert new_locdata.properties == selection.properties
    assert new_locdata.indices == selection.indices
    assert new_locdata.references != selection.references
    assert new_locdata.meta != selection.meta
    assert int(new_locdata.references.meta.identifier) == int(selection.meta.identifier) + 1
    assert int(new_locdata.meta.identifier) == int(selection.meta.identifier) + 2
    assert new_locdata.meta.creation_date == selection.meta.creation_date
