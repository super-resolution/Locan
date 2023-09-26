import tempfile
from pathlib import Path

import numpy as np
from google.protobuf import json_format
from pandas.testing import assert_frame_equal

import locan.constants
from locan import FileType
from locan.locan_io.locdata import manifest_pb2
from locan.locan_io.locdata.smlm_io import (
    _change_upper_to_lower_keys,
    load_SMLM_file,
    load_SMLM_header,
    load_SMLM_manifest,
    manifest_file_info_from_locdata,
    manifest_format_from_locdata,
    manifest_from_locdata,
    save_SMLM,
)


def test__change_upper_to_lower_keys():
    assert "text" == _change_upper_to_lower_keys("TEXT")


def test_manifest_format_from_locdata(locdata_2d):
    format = manifest_format_from_locdata(locdata_2d)
    assert isinstance(format, manifest_pb2.Format)
    # print(format)


def test_manifest_file_info_from_locdata(locdata_2d):
    file_info = manifest_file_info_from_locdata(locdata_2d)
    assert isinstance(file_info, manifest_pb2.FileInfo)
    # print(file_info)


def test_manifest_from_locdata(locdata_2d):
    manifest = manifest_from_locdata(locdata_2d)
    assert isinstance(manifest, manifest_pb2.Manifest)
    # print(manifest)

    # check for capital letters in manifest that should be introduced by manifest_pb2.Manifest
    json_string = json_format.MessageToJson(
        manifest, preserving_proto_field_name=True, including_default_value_fields=False
    )
    assert json_string
    assert "BINARY" in json_string
    assert "INT" in json_string

    # check that capital letters are eliminated by _change_upper_to_lower_keys in manifest_from_locdata
    manifest = manifest_from_locdata(locdata_2d, return_json_string=True)
    assert isinstance(manifest, str)
    assert "binary" in manifest
    assert "BINARY" not in manifest
    assert "int" in manifest
    assert "INT" not in manifest
    # print(manifest)


def test_get_correct_column_names_from_SMLM_header():
    columns = load_SMLM_header(
        path=locan.ROOT_DIR / "tests/test_data/SMLM_dstorm_data.smlm"
    )
    assert columns == [
        "original_index",
        "position_x",
        "local_background",
        "chi_square",
        "intensity",
        "frame",
        "position_y",
    ]


def test_load_SMLM_manifest():
    manifest = load_SMLM_manifest(
        path=locan.ROOT_DIR / "tests/test_data/SMLM_dstorm_data.smlm"
    )
    for key in ["format_version", "formats", "files"]:
        assert key in manifest.keys()


def test_loading_SMLM_file():
    file_path = locan.ROOT_DIR / "tests/test_data/SMLM_dstorm_data.smlm"
    locdata = load_SMLM_file(path=file_path, nrows=10)
    assert np.array_equal(
        locdata.data.columns,
        [
            "original_index",
            "position_x",
            "local_background",
            "chi_square",
            "intensity",
            "frame",
            "position_y",
        ],
    )
    assert len(locdata) == 10
    # print(locdata.data)
    assert locdata.meta.file.type == FileType.SMLM.value
    assert locdata.meta.file.path == str(file_path)


def test_save_and_load_smlm(locdata_2d):
    # introduce different dtypes in data
    locdata_2d.data["position_x"] = locdata_2d.data["position_x"].astype("float64")
    locdata_2d.data["position_y"] = locdata_2d.data["position_y"].astype("int32")
    locdata_2d.data["frame"] = locdata_2d.data["frame"].astype("int32")
    locdata_2d.data["intensity"] = locdata_2d.data["intensity"].astype("int32")
    assert all(locdata_2d.data.dtypes.values == ["float64", "int32", "int32", "int32"])
    # print(locdata_2d.data)
    # print(locdata_2d.data.dtypes)

    with tempfile.TemporaryDirectory() as tmp_directory:
        # save smlm file
        file_path = Path(tmp_directory) / "locdata.smlm"
        save_SMLM(locdata_2d, path=file_path)

        # read back smlm manifest
        manifest = load_SMLM_manifest(path=file_path)
        for key in ["format_version", "formats", "files"]:
            assert key in manifest.keys()

        # read back smlm file
        locdata = load_SMLM_file(path=file_path, convert=False)
        assert len(locdata) == len(locdata_2d)
        # assert (locdata.meta.identifier == locdata_2d.meta.identifier)
        assert locdata.properties.keys() == locdata_2d.properties.keys()

        # print(locdata.data)
        # print(locdata.data.dtypes)
        # print(manifest_from_locdata(locdata_2d))
        assert_frame_equal(locdata_2d.data, locdata.data, check_dtype=True)

        # passing manifest
        manifest = manifest_from_locdata(locdata_2d, return_json_string=True)
        save_SMLM(locdata_2d, path=file_path, manifest=manifest)

        manifest = manifest_from_locdata(locdata_2d)
        save_SMLM(locdata_2d, path=file_path, manifest=manifest)

        # read back smlm manifest
        manifest = load_SMLM_manifest(path=file_path)
        for key in ["format_version", "formats", "files"]:
            assert key in manifest.keys()

        # read back smlm file
        locdata = load_SMLM_file(path=file_path)
        assert_frame_equal(locdata_2d.data, locdata.data, check_dtype=False)
