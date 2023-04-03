import numpy as np
import pandas as pd
import pytest

from locan.data.locdata_utils import (
    _bump_property_label,
    _check_loc_properties,
    _get_available_coordinate_labels,
    _get_linked_coordinates,
)


@pytest.fixture()
def df_only_coordinates():
    return pd.DataFrame.from_dict(
        {"position_y": np.arange(3), "position_z": np.arange(3)}
    )


@pytest.fixture()
def df_with_zero_uncertainty():
    return pd.DataFrame.from_dict(
        {
            "position_y": np.arange(3),
            "uncertainty": np.arange(3),
            "position_z": np.arange(3),
            "uncertainty_z": np.arange(3),
        }
    )


@pytest.fixture()
def df_with_uncertainty():
    return pd.DataFrame.from_dict(
        {
            "position_y": np.arange(1, 3),
            "uncertainty": np.arange(1, 3),
            "position_z": np.arange(1, 3),
            "uncertainty_z": np.arange(1, 3),
        }
    )


@pytest.fixture()
def df_empty():
    return pd.DataFrame()


@pytest.fixture()
def df_single():
    return pd.DataFrame.from_dict(
        {
            "position_y": [1],
            "position_z": [2],
            "uncertainty_z": [5],
        }
    )


def test__get_available_coordinate_labels(df_only_coordinates, df_with_uncertainty):
    results = _get_available_coordinate_labels(locdata=df_only_coordinates)
    assert results == ([None, "position_y", "position_z"], [None, None, None])

    results = _get_available_coordinate_labels(locdata=df_with_uncertainty)
    assert results == (
        [None, "position_y", "position_z"],
        [None, "uncertainty", "uncertainty_z"],
    )

    results = _get_available_coordinate_labels(
        locdata=df_only_coordinates, coordinate_labels="position_y"
    )
    assert results == ([None, "position_y", None], [None, None, None])

    results = _get_available_coordinate_labels(
        locdata=df_with_uncertainty, coordinate_labels="position_y"
    )
    assert results == ([None, "position_y", None], [None, "uncertainty", None])

    results = _get_available_coordinate_labels(
        locdata=df_only_coordinates, coordinate_labels="position_z"
    )
    assert results == ([None, None, "position_z"], [None, None, None])

    results = _get_available_coordinate_labels(
        locdata=df_with_uncertainty, coordinate_labels="position_z"
    )
    assert results == ([None, None, "position_z"], [None, None, "uncertainty_z"])

    results = _get_available_coordinate_labels(
        locdata=df_only_coordinates, coordinate_labels="position_x"
    )
    assert results == ([None, None, None], [None, None, None])

    results = _get_available_coordinate_labels(
        locdata=df_with_uncertainty, coordinate_labels="position_x"
    )
    assert results == ([None, None, None], [None, None, None])

    results = _get_available_coordinate_labels(
        locdata=df_only_coordinates, coordinate_labels=["position_x", "position_z"]
    )
    assert results == ([None, None, "position_z"], [None, None, None])

    results = _get_available_coordinate_labels(
        locdata=df_with_uncertainty, coordinate_labels=["position_x", "position_z"]
    )
    assert results == ([None, None, "position_z"], [None, None, "uncertainty_z"])

    results = _get_available_coordinate_labels(
        locdata=df_only_coordinates, coordinate_labels=["position_y", "position_z"]
    )
    assert results == ([None, "position_y", "position_z"], [None, None, None])

    results = _get_available_coordinate_labels(
        locdata=df_with_uncertainty, coordinate_labels=["position_y", "position_z"]
    )
    assert results == (
        [None, "position_y", "position_z"],
        [None, "uncertainty", "uncertainty_z"],
    )


def test__get_linked_coordinates(
    df_only_coordinates,
    df_with_zero_uncertainty,
    df_with_uncertainty,
    df_empty,
    df_single,
):
    results = _get_linked_coordinates(locdata=df_only_coordinates)
    assert results == {
        "position_y": 1.0,
        "uncertainty_y": 0.3333333333333333,
        "position_z": 1.0,
        "uncertainty_z": 0.3333333333333333,
    }

    results = _get_linked_coordinates(locdata=df_with_zero_uncertainty)
    assert all(np.isnan(results[key]) for key in results.keys())

    results = _get_linked_coordinates(locdata=df_with_uncertainty)
    assert results == {
        "position_y": 1.3333333333333333,
        "uncertainty_y": 0.19753086419753085,
        "position_z": 1.3333333333333333,
        "uncertainty_z": 0.19753086419753085,
    }

    results = _get_linked_coordinates(
        locdata=df_only_coordinates, coordinate_labels="position_y"
    )
    assert results == {"position_y": 1.0, "uncertainty_y": 0.3333333333333333}

    results = _get_linked_coordinates(
        locdata=df_with_uncertainty, coordinate_labels="position_y"
    )
    assert results == {
        "position_y": 1.3333333333333333,
        "uncertainty_y": 0.19753086419753085,
    }

    results = _get_linked_coordinates(
        locdata=df_only_coordinates, coordinate_labels="position_z"
    )
    assert results == {"position_z": 1.0, "uncertainty_z": 0.3333333333333333}

    results = _get_linked_coordinates(
        locdata=df_with_uncertainty, coordinate_labels="position_z"
    )
    assert results == {
        "position_z": 1.3333333333333333,
        "uncertainty_z": 0.19753086419753085,
    }

    results = _get_linked_coordinates(locdata=df_empty)
    assert results == {}

    results = _get_linked_coordinates(locdata=df_single)
    assert results == {
        "position_y": 1,
        "uncertainty_y": 0,
        "position_z": 2,
        "uncertainty_z": 5,
    }


def test__bump_property_label():
    new_property = _bump_property_label(
        loc_properties=["test", "other_test"],
        loc_property="test",
        extension="_extended",
    )
    assert new_property == "test_extended"

    new_property = _bump_property_label(
        loc_properties=["test", "test_0"],
        loc_property="test",
    )
    assert new_property == "test_0_0"


def test__check_loc_properties(locdata_2d):
    with pytest.raises(ValueError):
        _check_loc_properties(locdata=locdata_2d, loc_properties=["test"])

    result = _check_loc_properties(locdata=locdata_2d, loc_properties="position_x")
    assert result == ["position_x"]

    result = _check_loc_properties(locdata=locdata_2d, loc_properties=None)
    assert result == ["position_x", "position_y"]

    result = _check_loc_properties(
        locdata=locdata_2d, loc_properties=locdata_2d.data.columns
    )
    assert result == locdata_2d.data.columns.tolist()
