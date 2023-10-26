import copy

import numpy as np
import pandas as pd
import pytest

from locan import AlphaShape, LocData, Rectangle, Region
from locan.data import metadata_pb2

# fixtures for DataFrames (fixtures for LocData are defined in conftest.py)


@pytest.fixture()
def df_simple():
    dict_ = {"position_x": [0, 0, 1, 4, 5], "position_y": [0, 1, 3, 4, 1]}
    return pd.DataFrame.from_dict(dict_)


@pytest.fixture()
def df_line():
    dict_ = {"position_x": [1, 2, 3, 4, 5], "position_y": [1, 2, 3, 4, 5]}
    return pd.DataFrame.from_dict(dict_)


@pytest.fixture()
def df_empty():
    dict_ = {}
    return pd.DataFrame.from_dict(dict_)


@pytest.fixture()
def df_other_simple():
    dict_ = {"position_x": [0, 0, 1, 4, 5], "position_y": [10, 11, 13, 14, 11]}
    return pd.DataFrame.from_dict(dict_)


# tests

COMMENT_METADATA = metadata_pb2.Metadata(comment="some user comment")


def test_LocData(df_simple, caplog):
    dat = LocData(dataframe=df_simple, meta=COMMENT_METADATA)
    assert len(dat) == 5
    assert dat.coordinate_keys == ["position_x", "position_y"]
    assert dat.dimension == 2
    assert np.array_equal(dat.centroid, [2.0, 1.8])
    assert dat.centroid[0] == dat.properties["position_x"]
    for x, y in zip(dat.coordinates, [[0, 0], [0, 1], [1, 3], [4, 4], [5, 1]]):
        assert np.all(x == np.array(y))
    assert dat.meta.comment == COMMENT_METADATA.comment
    # The following test runs ok when testing this function alone.
    # assert dat.meta.identifier == '1'
    assert dat.bounding_box.region_measure == 20
    assert "region_measure_bb" in dat.properties
    assert "localization_density_bb" in dat.properties
    assert dat.convex_hull.region_measure == 12.5
    assert "region_measure_ch" in dat.properties
    assert "localization_density_ch" in dat.properties
    assert round(dat.oriented_bounding_box.region_measure) == 16
    assert "region_measure_obb" in dat.properties
    assert "localization_density_obb" in dat.properties
    assert dat.region is None
    dat.region = Rectangle()
    assert caplog.record_tuples == [
        ("locan.data.locdata", 30, "Not all coordinates are within region.")
    ]
    dat.region = dat.bounding_box.region
    assert repr(dat.region) == repr(dat.bounding_box.region)
    assert isinstance(dat.update_alpha_shape(alpha=1).alpha_shape, AlphaShape)
    assert "region_measure_as" in dat.properties
    assert "localization_density_as" in dat.properties

    assert len(dat.inertia_moments.eigenvalues) == 2
    assert "orientation_im" in dat.properties
    assert "circularity_im" in dat.properties


def test_update_properties():
    df_with_coordinates = pd.DataFrame.from_dict(
        {
            "position_y": np.arange(1, 3),
        }
    )

    df_with_all = pd.DataFrame.from_dict(
        {
            "position_y": np.arange(1, 3),
            "uncertainty": np.arange(1, 3),
            "intensity": np.arange(1, 3),
            "frame": np.arange(1, 3),
        }
    )

    locdata = LocData.from_dataframe(df_with_coordinates)
    assert locdata.properties == {
        "localization_count": 2,
        "position_y": 1.5,
        "uncertainty_y": 0.5,
        "region_measure_bb": 1,
        "localization_density_bb": 2.0,
        "subregion_measure_bb": 2,
    }

    locdata = LocData.from_dataframe(df_with_all)
    assert locdata.properties == {
        "localization_count": 2,
        "position_y": 1.2,
        "uncertainty_y": 0.32,
        "intensity": 3,
        "frame": 1,
        "region_measure_bb": 1,
        "localization_density_bb": 2.0,
        "subregion_measure_bb": 2,
    }

    locdata = LocData.from_dataframe(df_with_all)
    update_function = {"position_y": np.mean}
    locdata = locdata._update_properties(update_function=update_function)
    assert locdata.properties == {
        "localization_count": 2,
        "intensity": 3,
        "frame": 1,
        "position_y": 1.5,
        "region_measure_bb": 1,
        "localization_density_bb": 2.0,
        "subregion_measure_bb": 2,
    }


def test_LocData_empty(df_empty):
    dat = LocData()
    assert dat.data.empty
    assert dat.properties == {"localization_count": 0, "region_measure_bb": 0}
    assert len(dat) == 0
    assert dat.coordinate_keys == []
    assert dat.dimension == 0

    dat = LocData(dataframe=df_empty)
    assert dat.data.empty
    assert dat.properties == {"localization_count": 0, "region_measure_bb": 0}
    assert len(dat) == 0
    assert dat.coordinate_keys == []
    assert dat.dimension == 0
    # hulls are tested with locdata fixtures further down.


def test_LocData_from_dataframe(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    # print(dat.properties.keys())
    assert dat.bounding_box is not None
    assert list(dat.properties.keys()) == [
        "localization_count",
        "position_x",
        "uncertainty_x",
        "position_y",
        "uncertainty_y",
        "region_measure_bb",
        "localization_density_bb",
        "subregion_measure_bb",
    ]
    assert len(dat) == 5
    assert dat.meta.comment == COMMENT_METADATA.comment

    dat = LocData.from_dataframe(dataframe=df_simple.__dataframe__())
    assert len(dat) == 5

    dat = LocData.from_dataframe(dataframe=None)
    assert len(dat) == 0


def test_LocData_from_dataframe_empty(df_empty):
    dat = LocData.from_dataframe(dataframe=df_empty)
    assert len(dat) == 0
    assert dat.coordinate_keys == []
    # print(dat.data)


def test_LocData_from_dataframe_with_meta_dict(df_simple):
    dat = LocData.from_dataframe(
        dataframe=df_simple, meta={"comment": "some user comment"}
    )
    assert len(dat) == 5
    assert dat.references is None
    assert dat.meta.comment == COMMENT_METADATA.comment


def test_LocData_from_coordinates():
    coordinates = [(200, 500), (200, 600), (900, 650), (1000, 600)]
    dat = LocData.from_coordinates(coordinates=coordinates, meta=COMMENT_METADATA)
    assert np.array_equal(dat.coordinates, np.asarray(coordinates))
    assert all(item in ["position_x", "position_y"] for item in dat.coordinate_keys)
    assert len(dat) == 4
    assert dat.meta.comment == COMMENT_METADATA.comment

    coordinates = np.array([(200, 500), (200, 600), (900, 650), (1000, 600)])
    dat = LocData.from_coordinates(coordinates=coordinates, meta=COMMENT_METADATA)
    assert np.array_equal(dat.coordinates, np.asarray(coordinates))
    assert all(item in ["position_x", "position_y"] for item in dat.coordinate_keys)
    assert len(dat) == 4

    dat = LocData.from_coordinates(
        coordinates=coordinates,
        coordinate_labels=["position_x", "position_z"],
        meta=dict(comment="special order"),
    )
    assert np.array_equal(dat.coordinates, np.asarray(coordinates))
    assert all(item in ["position_x", "position_z"] for item in dat.coordinate_keys)
    assert len(dat) == 4

    with pytest.raises(ValueError):
        LocData.from_coordinates(
            coordinates=coordinates, coordinate_labels=["position_x", "whatever"]
        )

    dat = LocData.from_coordinates(coordinates=[])
    assert len(dat) == 0


def test_LocData_from_selection(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple)

    sel = LocData.from_selection(locdata=dat, indices=slice(1, 3))
    assert all(sel.data.index == [1, 2, 3])

    sel = LocData.from_selection(locdata=dat)
    assert len(sel) == len(df_simple)

    sel = LocData.from_selection(locdata=dat, indices=slice(4, 10))
    assert all(sel.data.index == [4])

    sel = LocData.from_selection(locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA)
    assert len(sel) == 3
    assert sel.references is dat
    assert sel.meta.comment == COMMENT_METADATA.comment
    assert dat.bounding_box.region_measure != sel.bounding_box.region_measure
    with pytest.raises(AttributeError):
        sel.data = df_simple

    sel_sel = LocData.from_selection(
        locdata=sel, indices=[1, 3], meta={"comment": "Selection of a selection."}
    )
    assert sel_sel.meta.comment == "Selection of a selection."
    assert len(sel_sel) == 2
    assert sel_sel.references is sel

    sel_empty = LocData.from_selection(
        locdata=dat, indices=[], meta={"comment": "This selection is empty."}
    )
    assert sel_empty.data.empty
    assert len(sel_empty) == 0

    dat = LocData()
    sel_empty = LocData.from_selection(locdata=dat, indices=[])
    assert sel_empty.data.empty
    assert len(sel_empty) == 0

    sel_empty = LocData.from_selection(locdata=dat, indices=[1, 3])
    assert sel_empty.data.empty
    assert len(sel_empty) == 0


def test_LocData_from_collection(df_simple):
    col = LocData.from_collection([LocData(), LocData()], meta=COMMENT_METADATA)
    assert col.dimension == 0
    assert len(col) == 2
    assert col.properties == {"localization_count": 2, "region_measure_bb": 0}
    assert col.meta.comment == COMMENT_METADATA.comment

    dat = LocData.from_dataframe(dataframe=df_simple)
    sel_1 = LocData.from_selection(locdata=dat, indices=[0, 1, 2])
    sel_2 = LocData.from_selection(locdata=dat, indices=[3, 4])
    col = LocData.from_collection(
        [sel_1, sel_2], meta=dict(comment="yet another comment")
    )
    assert len(col.references) == 2
    assert len(col) == 2
    assert col.meta.comment == "yet another comment"
    assert set(col.data.columns) == {
        "localization_count",
        "localization_density_bb",
        "position_x",
        "position_y",
        "region_measure_bb",
        "subregion_measure_bb",
        "uncertainty_x",
        "uncertainty_y",
    }

    with pytest.warns(UserWarning):
        col.update_convex_hulls_in_references()
    assert set(col.data.columns) == {
        "localization_count",
        "localization_density_bb",
        "position_x",
        "position_y",
        "region_measure_bb",
        "subregion_measure_bb",
        "region_measure_ch",
        "subregion_measure_ch",
        "localization_density_ch",
        "uncertainty_x",
        "uncertainty_y",
    }

    col.update_oriented_bounding_box_in_references()
    assert set(col.data.columns) == {
        "circularity_obb",
        "localization_count",
        "localization_density_bb",
        "localization_density_ch",
        "localization_density_obb",
        "orientation_obb",
        "position_x",
        "position_y",
        "region_measure_bb",
        "region_measure_ch",
        "region_measure_obb",
        "subregion_measure_bb",
        "subregion_measure_ch",
        "uncertainty_x",
        "uncertainty_y",
    }
    # print(col.properties)

    with pytest.warns(UserWarning):
        col.update_alpha_shape_in_references(alpha=1)
    assert set(col.data.columns) == {
        "circularity_obb",
        "localization_count",
        "localization_density_bb",
        "localization_density_ch",
        "localization_density_obb",
        "orientation_obb",
        "position_x",
        "position_y",
        "region_measure_bb",
        "region_measure_ch",
        "region_measure_obb",
        "subregion_measure_bb",
        "subregion_measure_ch",
        "region_measure_as",
        "localization_density_as",
        "uncertainty_x",
        "uncertainty_y",
    }

    col.update_inertia_moments_in_references()
    assert set(col.data.columns) == {
        "circularity_obb",
        "localization_count",
        "localization_density_bb",
        "localization_density_ch",
        "localization_density_obb",
        "orientation_obb",
        "position_x",
        "position_y",
        "region_measure_bb",
        "region_measure_ch",
        "region_measure_obb",
        "subregion_measure_bb",
        "subregion_measure_ch",
        "region_measure_as",
        "localization_density_as",
        "circularity_im",
        "orientation_im",
        "uncertainty_x",
        "uncertainty_y",
    }


def test_LocData_selection_from_collection(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple)
    sel = []
    col = LocData.from_collection(sel)
    assert len(col) == 0

    for i in range(4):
        sel.append(LocData.from_selection(locdata=dat, indices=[i]))
    col = LocData.from_collection(sel, meta=COMMENT_METADATA)
    assert len(col) == 4
    assert len(col.references) == 4
    # print(col.data)

    col_sel = LocData.from_selection(locdata=col, indices=[0, 2, 3])
    assert len(col_sel) == 3
    assert col_sel.references is col
    # print(col_sel.data)

    col_sel_sel = LocData.from_selection(locdata=col_sel, indices=[2])
    assert len(col_sel_sel) == 1
    assert col_sel_sel.references is col_sel
    # print(col_sel_sel.data)


def test_LocData_concat(df_simple):
    col = LocData.concat([LocData(), LocData()], meta=COMMENT_METADATA)
    assert col.dimension == 0
    assert len(col) == 0
    assert col.properties == {"localization_count": 0, "region_measure_bb": 0}
    assert col.meta.comment == COMMENT_METADATA.comment

    dat = LocData.from_dataframe(dataframe=df_simple)
    sel_1 = LocData.from_selection(locdata=dat, indices=[0, 1, 2])
    sel_2 = LocData.from_selection(locdata=dat, indices=[3, 4])
    col = LocData.concat([sel_1, sel_2], meta=dict(comment="yet another comment"))
    assert len(col) == 5
    assert col.meta.comment == "yet another comment"
    assert len(col.references) == 2
    sel_1.reduce()
    col = LocData.concat([sel_1, sel_2], meta=COMMENT_METADATA)
    assert len(col.references) == 2
    sel_2.reduce()
    col = LocData.concat([sel_1, sel_2], meta=COMMENT_METADATA)
    assert col.references is None


def test_LocData_reduce(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    sel_1 = LocData.from_selection(
        locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA
    )
    sel_2 = LocData.from_selection(
        locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA
    )
    sel_1.reduce()
    assert sel_1.references is None
    assert len(sel_1) == 3
    assert len(sel_1.data) == len(sel_2.data)
    col = LocData.from_collection([sel_1, sel_2], meta=COMMENT_METADATA)
    col.reduce()
    assert len(col) == 2
    assert col.references is None


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("locdata_empty", 0), ("locdata_single_localization", 1)],
)
def test_locdata_reduce_empty(
    locdata_empty, locdata_single_localization, fixture_name, expected
):
    dat = eval(fixture_name)
    new_dat = dat.reduce()
    assert len(new_dat) == expected


def test_LocData_reset(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    assert dat.properties["position_x"] == 2
    assert dat.properties["region_measure_bb"] == 20
    assert dat.meta.comment == COMMENT_METADATA.comment

    dat.dataframe["position_x"] = [0, 1, 4, 0, 0]
    dat.dataframe["position_y"] = [1, 3, 4, 0, 0]
    dat.reset()
    assert dat.properties["position_x"] == 1
    assert dat.properties["region_measure_bb"] == 16
    assert dat.meta.comment == COMMENT_METADATA.comment


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("locdata_empty", 0), ("locdata_single_localization", 1)],
)
def test_locdata_reset_empty(
    locdata_empty, locdata_single_localization, fixture_name, expected
):
    dat = eval(fixture_name)
    new_dat = dat.reset()
    assert len(new_dat) == expected


def test_LocData_update(df_simple, caplog):
    new_dataframe = pd.DataFrame.from_dict(
        {
            "position_x": [10, 0, 1, 4],
            "position_y": [10, 1, 3, 4],
            "frame": [0, 1, 1, 4],
        }
    )

    dat = LocData()
    dat.update(new_dataframe, meta=COMMENT_METADATA)
    pd.testing.assert_frame_equal(dat.data, new_dataframe)
    assert dat.properties["position_x"] == 3.75
    assert dat.meta.element_count == 4
    assert dat.meta.history[-1].name == "LocData.update"

    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    assert dat.properties["position_x"] == 2
    assert dat.meta.element_count == 5

    dat.update(new_dataframe, meta=dict(comment="yet another comment"))
    pd.testing.assert_frame_equal(dat.data, new_dataframe)
    assert dat.properties["position_x"] == 3.75
    assert dat.meta.element_count == 4
    assert dat.meta.history[-1].name == "LocData.update"

    sel = LocData.from_selection(locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA)
    sel.update(new_dataframe, reset_index=True)
    assert caplog.record_tuples.pop() == (
        "locan.data.locdata",
        30,
        "LocData.reduce() was applied since self.references was not None.",
    )
    pd.testing.assert_frame_equal(sel.data, new_dataframe)
    assert sel.properties["position_x"] == 3.75
    assert sel.meta.element_count == 4
    assert sel.meta.history[-1].name == "LocData.update"

    sel = LocData.from_selection(
        locdata=dat, indices=[1, 2, 3, 4], meta=COMMENT_METADATA
    )
    sel.update(new_dataframe, reset_index=True)
    assert caplog.record_tuples.pop() == (
        "locan.data.locdata",
        30,
        "LocData.reduce() was applied since self.references was not None.",
    )
    pd.testing.assert_frame_equal(sel.data, new_dataframe)
    assert sel.properties["position_x"] == 3.75
    assert sel.meta.element_count == 4
    assert sel.meta.history[-1].name == "LocData.update"


def test_LocData_projection(locdata_3d):
    locdata = copy.deepcopy(locdata_3d)

    # test with single coordinate_label
    new_coordinate_labels = "position_x"
    new_locdata = locdata.projection(coordinate_labels=new_coordinate_labels)

    assert all(label in new_coordinate_labels for label in new_locdata.coordinate_keys)
    assert new_locdata.dimension == 1
    assert len(new_locdata) == len(locdata)

    # test with multiple coordinate_properties
    new_coordinate_labels = ["position_x", "position_y"]
    new_locdata = locdata.projection(coordinate_labels=new_coordinate_labels)

    assert all(label in new_coordinate_labels for label in new_locdata.coordinate_keys)
    assert new_locdata.dimension == len(new_coordinate_labels)
    assert len(new_locdata) == len(locdata)


# locdata with added columns


def test_LocData_add_column_to_dataframe(df_simple):
    # from dataframe
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    dat.dataframe = dat.dataframe.assign(new=np.arange(5))
    assert dat.data.equals(dat.dataframe)

    # from selection
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    sel = LocData.from_selection(locdata=dat, indices=[0, 1, 2])
    sel.dataframe = sel.dataframe.assign(new=np.arange(3))
    assert list(sel.dataframe.columns) == ["new"]
    assert list(sel.references.dataframe.columns) == ["position_x", "position_y"]
    assert all(list(sel.data.columns == ["position_x", "position_y", "new"]))
    sel.reduce()
    assert all(list(sel.dataframe.columns == ["position_x", "position_y", "new"]))

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
    assert all(
        list(
            col.data.columns
            == [
                "localization_count",
                "position_x",
                "uncertainty_x",
                "position_y",
                "uncertainty_y",
                "region_measure_bb",
                "localization_density_bb",
                "subregion_measure_bb",
                "new",
            ]
        )
    )


# locdata and metadata


def test_LocData_handling_metadata(df_simple):
    dat = LocData.from_dataframe(dataframe=df_simple, meta=COMMENT_METADATA)
    # print(dat.meta)
    # print(dir(metadata_pb2))
    assert dat.meta.identifier
    assert dat.meta.creation_time
    assert dat.meta.source == metadata_pb2.DESIGN
    assert dat.meta.state == metadata_pb2.RAW
    assert dat.meta.history[0].name == "LocData.from_dataframe"
    assert dat.meta.element_count == 5

    dat.meta.comment = "new comment"
    assert dat.meta.comment == "new comment"

    dat.meta.map["variable key"] = "new comment"
    assert dict(dat.meta.map) == {"variable key": "new comment"}

    dat.meta.map["key_2"] = "value_2"
    assert dat.meta.map["key_2"] == "value_2"

    dat = LocData.from_dataframe(
        dataframe=df_simple, meta=dict(comment="yet another comment")
    )
    assert dat.meta.comment == "yet another comment"


# locdata and regions


def test_locdata_region(df_simple):
    region = Rectangle((0, 0), 2, 1, 10)
    dat = LocData.from_dataframe(dataframe=df_simple)
    dat.region = region
    assert isinstance(dat.region, Region)
    assert dat.region.region_measure == 2


# standard LocData fixtures


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("locdata_empty", 0),
        ("locdata_single_localization", 1),
        ("locdata_2d", 6),
        ("locdata_non_standard_index", 6),
    ],
)
def test_standard_locdata_objects(
    locdata_empty,
    locdata_single_localization,
    locdata_2d,
    locdata_non_standard_index,
    fixture_name,
    expected,
):
    dat = eval(fixture_name)
    assert len(dat) == expected


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("locdata_empty", 0), ("locdata_single_localization", 0)],
)
def test_locdata_hulls(
    locdata_empty, locdata_single_localization, fixture_name, expected
):
    dat = eval(fixture_name)
    assert dat.bounding_box.region_measure == 0
    assert dat.oriented_bounding_box.region_measure == 0
    with pytest.warns(UserWarning):
        assert dat.convex_hull is None
    with pytest.raises(AttributeError):
        assert dat.region_measure_ch
    if len(dat) == 0:
        assert isinstance(dat.update_alpha_shape(alpha=1).alpha_shape, AlphaShape)
    else:
        with pytest.warns(UserWarning):
            assert isinstance(dat.update_alpha_shape(alpha=1).alpha_shape, AlphaShape)


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("locdata_empty", 0), ("locdata_single_localization", 0)],
)
def test_locdata_from_selection_exceptions(
    locdata_empty,
    locdata_single_localization,
    locdata_2d,
    locdata_non_standard_index,
    fixture_name,
    expected,
):
    dat = eval(fixture_name)
    new_dat = LocData.from_selection(
        locdata=dat, indices=[1, 3, 4], meta=COMMENT_METADATA
    )
    assert len(new_dat) == expected


@pytest.mark.parametrize(
    "fixture_name, expected", [("locdata_2d", 3), ("locdata_non_standard_index", 3)]
)
def test_locdata_from_selection_(
    locdata_empty,
    locdata_single_localization,
    locdata_2d,
    locdata_non_standard_index,
    fixture_name,
    expected,
):
    dat = eval(fixture_name)
    sel = LocData.from_selection(locdata=dat, indices=[1, 2, 5], meta=COMMENT_METADATA)
    assert len(sel) == expected
    assert set(sel.data.index) == set([1, 2, 5])
    # in coming versions: assert list(sel.data.index) == [2, 1, 5]
    assert sel.references is dat
    assert sel.meta.comment == COMMENT_METADATA.comment


def test_locdata_from_chunks_(locdata_non_standard_index):
    with pytest.raises(ValueError):
        chunk_collection = LocData.from_chunks(locdata=locdata_non_standard_index)

    chunk_collection = LocData.from_chunks(
        locdata=locdata_non_standard_index, chunks=((2, 1), (0, 3))
    )
    assert all(chunk_collection.references[0].data.index == [2, 1])

    chunk_collection = LocData.from_chunks(
        locdata=locdata_non_standard_index, chunk_size=2
    )
    assert all(chunk_collection.references[0].data.index == [2, 1])

    chunk_collection = LocData.from_chunks(
        locdata=locdata_non_standard_index, n_chunks=3
    )
    assert all(chunk_collection.references[0].data.index == [2, 1])

    chunk_collection = LocData.from_chunks(
        locdata=locdata_non_standard_index, n_chunks=3, order="alternating"
    )
    assert all(chunk_collection.references[0].data.index == [2, 10])


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("locdata_empty", (0, (0,), 0, (0,))),
        ("locdata_single_localization", (1, (1,), 1, (1,))),
        ("locdata_2d", (3, (2, 2, 2), 1, (4,))),
        ("locdata_non_standard_index", (3, (2, 2, 2), 1, (4,))),
    ],
)
def test_locdata_from_chunks(
    locdata_empty,
    locdata_single_localization,
    locdata_2d,
    locdata_non_standard_index,
    fixture_name,
    expected,
):
    dat = eval(fixture_name)
    chunk_collection = LocData.from_chunks(
        locdata=dat, chunk_size=2, meta=COMMENT_METADATA
    )
    assert isinstance(chunk_collection.references, list)
    assert len(chunk_collection) == expected[0]
    if len(chunk_collection) != 0:
        assert all(chunk_collection.data.localization_count == expected[1])
    assert chunk_collection.meta.comment == COMMENT_METADATA.comment

    chunk_collection = LocData.from_chunks(
        locdata=dat,
        chunk_size=2,
        order="alternating",
        meta=dict(comment="yet another comment"),
    )
    assert isinstance(chunk_collection.references, list)
    assert len(chunk_collection) == expected[0]
    if len(chunk_collection) != 0:
        assert all(chunk_collection.data.localization_count == expected[1])
    assert chunk_collection.meta.comment == "yet another comment"

    chunk_collection = LocData.from_chunks(locdata=dat, chunk_size=4, drop=True)
    assert isinstance(chunk_collection.references, list)
    assert len(chunk_collection) == expected[2]
    if len(chunk_collection) != 0:
        assert all(chunk_collection.data.localization_count == expected[3])


def test_copy():
    locdata = LocData.from_dataframe(
        dataframe=pd.DataFrame({"col_1": [1, 2, 3], "col_2": ["a", "b", "c"]})
    )
    selection = LocData.from_selection(locdata, indices=[0, 1])
    new_locdata = copy.copy(selection)

    assert new_locdata is not selection
    assert len(selection) == len(new_locdata)
    for attr in ["dataframe", "indices", "references"]:
        assert getattr(selection, attr) is getattr(new_locdata, attr)
    for attr in ["properties", "data", "meta"]:
        assert getattr(selection, attr) is not getattr(new_locdata, attr)
    pd.testing.assert_frame_equal(new_locdata.dataframe, selection.dataframe)
    pd.testing.assert_frame_equal(new_locdata.data, selection.data)
    assert new_locdata.properties == selection.properties
    assert new_locdata.indices == selection.indices
    assert new_locdata.references == selection.references
    assert new_locdata.meta != selection.meta
    assert (
        int(new_locdata.references.meta.identifier)
        == int(selection.meta.identifier) - 1
    )
    assert int(new_locdata.meta.identifier) == int(selection.meta.identifier) + 1
    assert new_locdata.meta.creation_time == selection.meta.creation_time


def test_deepcopy():
    locdata = LocData.from_dataframe(
        dataframe=pd.DataFrame({"col_1": [1, 2, 3], "col_2": ["a", "b", "c"]})
    )
    selection = LocData.from_selection(locdata, indices=[0, 1])
    new_locdata = copy.deepcopy(selection)

    assert new_locdata is not selection
    assert len(selection) == len(new_locdata)
    for attr in ["data", "dataframe", "indices", "meta", "properties", "references"]:
        assert getattr(selection, attr) is not getattr(new_locdata, attr)
    pd.testing.assert_frame_equal(new_locdata.dataframe, selection.dataframe)
    pd.testing.assert_frame_equal(new_locdata.data, selection.data)
    assert new_locdata.properties == selection.properties
    assert new_locdata.indices == selection.indices
    assert new_locdata.references != selection.references
    assert new_locdata.meta != selection.meta
    assert (
        int(new_locdata.references.meta.identifier)
        == int(selection.meta.identifier) + 1
    )
    assert int(new_locdata.meta.identifier) == int(selection.meta.identifier) + 2
    assert new_locdata.meta.creation_time == selection.meta.creation_time


def test_locdata_print_summary(capfd, locdata_2d):
    locdata_2d.print_summary()
    captured = capfd.readouterr()
    for element in ["identifier", "creation_time", "source", "state", "element_count"]:
        assert element in captured.out


def test_locdata_print_meta(capfd, locdata_2d):
    locdata_2d.print_meta()
    captured = capfd.readouterr()
    for element in ["identifier", "creation_time", "source", "state", "element_count"]:
        assert element in captured.out


def test_locdata_coordinate_labels(locdata_2d):
    c_labels = locdata_2d.coordinate_keys
    assert c_labels == ["position_x", "position_y"]
    c_labels.append("other")
    assert c_labels == ["position_x", "position_y", "other"]
    c_labels = locdata_2d.coordinate_keys
    assert c_labels == ["position_x", "position_y"]


def test_locdata_uncertainty_labels(locdata_2d):
    c_labels = locdata_2d.uncertainty_keys
    assert c_labels == []
    c_labels.append("other")
    assert c_labels == ["other"]
    c_labels = locdata_2d.uncertainty_keys
    assert c_labels == []


def test_update_properties_in_references(df_simple, caplog):
    locdata = LocData(dataframe=df_simple)
    other_locdata = LocData(dataframe=df_simple)
    collection = LocData.from_collection([locdata, other_locdata])
    collection.dataframe.index = [1, 3]

    # test dict
    new_collection = copy.deepcopy(collection)

    new_properties = {"new_property": [111, 222]}

    new_collection.update_properties_in_references(properties=new_properties)

    for reference_ in new_collection.references:
        assert "new_property" in reference_.properties
    assert new_collection.data.index.tolist() == collection.data.index.tolist()
    assert "new_property" in new_collection.data.columns
    assert new_collection.data.new_property.tolist() == [111, 222]
    for key, value in collection.properties.items():
        assert value == new_collection.properties[key]

    # test Series with custom index
    new_collection = copy.deepcopy(collection)
    new_properties = pd.Series(name="new_property", data=[111, 222])
    new_properties.index = [3, 1]
    with pytest.raises(ValueError):
        new_collection.update_properties_in_references(properties=new_properties)

    # test DataFrame with custom index
    new_collection = copy.deepcopy(collection)
    new_properties = pd.DataFrame.from_dict(
        {"new_property": [111, 222], "new_property_2": [1111, 2222]}
    )
    new_properties.index = [3, 1]
    with pytest.raises(ValueError):
        new_collection.update_properties_in_references(properties=new_properties)
