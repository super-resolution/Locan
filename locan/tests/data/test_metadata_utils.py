from copy import copy

import google.protobuf.message
import pytest

from locan import (
    load_metadata_from_toml,
    message_scheme,
    metadata_from_toml_string,
    metadata_to_formatted_string,
)
from locan.data import metadata_pb2
from locan.data.metadata_utils import _dict_to_protobuf, _modify_meta


@pytest.fixture
def metadata():
    metadata = metadata_pb2.Metadata()
    optical_unit = metadata.experiment.setups.add().optical_units.add()
    optical_unit.detection.camera.integration_time.FromMilliseconds(20)

    metadata.identifier = "111"
    metadata.comment = "comment"
    metadata.creation_time.FromJsonString("1111-11-11T11:11:11+01:00")

    return metadata


@pytest.fixture
def meta_dict():
    meta_dict_ = {
        "identifier": "123",
        "comment": "my comment",
        "ancestor_identifiers": ["1", "2"],
        "production_time": "2022-05-14T06:58:00Z",
        "localizer": {"software": "rapidSTORM"},
        "relations": [{"identifier": "1"}],
        "experiment": {
            "setups": [
                {
                    "identifier": "1",
                    "optical_units": [
                        {
                            "identifier": "1",
                            "detection": {
                                "camera": {
                                    "identifier": "1",
                                    "integration_time": 10000000,
                                }
                            },
                        }
                    ],
                }
            ]
        },
    }
    return meta_dict_


@pytest.fixture
def metadata_toml():
    metadata_toml_ = """
    # Define the class (message) instances.

    [[messages]]
    name = "metadata"
    module = "locan.data.metadata_pb2"
    class_name = "Metadata"


    # Fill metadata attributes
    # Use [[]] to add repeated elements
    # Use string '2022-05-14T06:58:00Z' for Timestamp elements
    # Use int in nanoseconds for Duration elements

    [metadata]
    identifier = "123"
    comment = "my comment"
    ancestor_identifiers = ["1", "2"]
    production_time = '2022-05-14T06:58:00Z'

    [metadata.localizer]
    software = "rapidSTORM"

    [[metadata.relations]]
    identifier = "1"

    [[metadata.experiment.setups]]
    identifier = "1"

    [[metadata.experiment.setups.optical_units]]
    identifier = "1"

    [metadata.experiment.setups.optical_units.detection.camera]
    identifier = "1"
    integration_time = 10_000_000
    """
    return metadata_toml_


def test__modify_meta(metadata, locdata_2d):
    new_locdata = copy(locdata_2d)
    new_metadata = _modify_meta(
        locdata_2d,
        new_locdata,
        function_name="test_function",
        parameter=dict(first=1, second=2),
        meta=None,
    )

    assert (
        metadata_to_formatted_string(message=new_metadata.history[-1], as_one_line=True)
        == "name: \"test_function\" parameter: \"{\\'first\\': 1, \\'second\\': 2}\""
    )


def test_metadata_to_formatted_string(metadata):
    results_string = metadata_to_formatted_string(message=metadata, as_one_line=True)
    results_string_expected = (
        'identifier: "111" comment: "comment" experiment { setups { optical_units { detection '
        "{ camera { integration_time { 0.020s } } } } } } creation_time { 1111-11-11T10:11:11Z }"
    )
    assert results_string == results_string_expected


def test__dict_to_protobuf(meta_dict):
    metadata = metadata_pb2.Metadata()
    metadata.comment = "to be changed"
    metadata.element_count = 1  # will not be touched
    new_metadata = _dict_to_protobuf(dictionary=meta_dict, message=metadata)
    assert isinstance(new_metadata, google.protobuf.message.Message)
    results_string = metadata_to_formatted_string(
        message=new_metadata, as_one_line=True
    )
    results_string_expected = (
        'identifier: "123" comment: "my comment" ancestor_identifiers: "1" '
        'ancestor_identifiers: "2" element_count: 1 relations { identifier: "1" } '
        'experiment { setups { identifier: "1" '
        'optical_units { identifier: "1" detection { '
        'camera { identifier: "1" integration_time { 0.010s } } } } } } '
        'localizer { software: "rapidSTORM" } production_time { 2022-05-14T06:58:00Z }'
    )
    assert results_string == results_string_expected

    metadata = metadata_pb2.Metadata()
    metadata.comment = "to be changed"
    metadata.element_count = 1  # will not be touched
    new_metadata = _dict_to_protobuf(
        dictionary=meta_dict, message=metadata, inplace=True
    )
    assert new_metadata is None
    results_string = metadata_to_formatted_string(message=metadata, as_one_line=True)
    results_string_expected = (
        'identifier: "123" comment: "my comment" ancestor_identifiers: "1" '
        'ancestor_identifiers: "2" element_count: 1 relations { identifier: "1" } '
        'experiment { setups { identifier: "1" '
        'optical_units { identifier: "1" detection { '
        'camera { identifier: "1" integration_time { 0.010s } } } } } } '
        'localizer { software: "rapidSTORM" } production_time { 2022-05-14T06:58:00Z }'
    )
    assert results_string == results_string_expected


def test_metadata_from_toml_string(metadata_toml):
    result = metadata_from_toml_string(metadata_toml)
    assert isinstance(result, dict)
    assert isinstance(result["metadata"], google.protobuf.message.Message)
    results_string = metadata_to_formatted_string(
        message=result["metadata"], as_one_line=True
    )
    results_string_expected = (
        'identifier: "123" comment: "my comment" ancestor_identifiers: "1" '
        'ancestor_identifiers: "2" relations { identifier: "1" } experiment { '
        'setups { identifier: "1" optical_units { identifier: "1" detection { '
        'camera { identifier: "1" integration_time { 0.010s } } } } } } '
        'localizer { software: "rapidSTORM" } production_time { 2022-05-14T06:58:00Z }'
    )
    assert results_string == results_string_expected


def test_metadata_from_toml_file(tmp_path, metadata_toml):
    file_path = tmp_path / "metadata_toml.toml"
    with file_path.open("w", encoding="utf-8") as file:
        file.write(metadata_toml)

    # load path-like
    result = load_metadata_from_toml(file_path)

    assert isinstance(result["metadata"], google.protobuf.message.Message)
    results_string = metadata_to_formatted_string(
        message=result["metadata"], as_one_line=True
    )
    results_string_expected = (
        'identifier: "123" comment: "my comment" ancestor_identifiers: "1" '
        'ancestor_identifiers: "2" relations { identifier: "1" } experiment { '
        'setups { identifier: "1" optical_units { identifier: "1" detection { '
        'camera { identifier: "1" integration_time { 0.010s } } } } } } '
        'localizer { software: "rapidSTORM" } production_time { 2022-05-14T06:58:00Z }'
    )
    assert results_string == results_string_expected

    # load file-like
    with open(file_path, "rb") as f:
        result = load_metadata_from_toml(f)
    assert isinstance(result, dict)


def test_message_scheme(metadata):
    scheme = message_scheme(metadata)
    assert isinstance(scheme, dict)
    assert scheme["creation_time"] == "1111-11-11T10:11:11Z"
    assert scheme["modification_time"] == "1970-01-01T00:00:00Z"
