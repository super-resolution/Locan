import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
from google.protobuf import json_format
from pandas.testing import assert_frame_equal

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
from tests import TEST_DIR


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
        manifest,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=False,
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
    columns = load_SMLM_header(path=TEST_DIR / "test_data/SMLM_dstorm_data.smlm")
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
    manifest = load_SMLM_manifest(path=TEST_DIR / "test_data/SMLM_dstorm_data.smlm")
    for key in ["format_version", "formats", "files"]:
        assert key in manifest.keys()


def test_loading_SMLM_file():
    file_path = TEST_DIR / "test_data/SMLM_dstorm_data.smlm"
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
    locdata = deepcopy(locdata_2d)
    # introduce different dtypes in data
    locdata.data["position_x"] = locdata.data["position_x"].astype("float64")
    locdata.data["position_y"] = locdata.data["position_y"].astype("int32")
    locdata.data["frame"] = locdata.data["frame"].astype("int32")
    locdata.data["intensity"] = locdata.data["intensity"].astype("int32")
    assert all(locdata.data.dtypes.values == ["float64", "int32", "int32", "int32"])
    # print(locdata.data)

    with tempfile.TemporaryDirectory() as tmp_directory:
        # save smlm file
        file_path = Path(tmp_directory) / "locdata.smlm"
        save_SMLM(locdata, path=file_path)

        # read back smlm manifest
        manifest = load_SMLM_manifest(path=file_path)
        for key in ["format_version", "formats", "files"]:
            assert key in manifest.keys()

        # read back smlm file
        locdata_ = load_SMLM_file(path=file_path, convert=False)
        assert len(locdata_) == len(locdata)
        assert locdata_.meta.identifier != locdata.meta.identifier
        assert locdata_.properties.keys() == locdata.properties.keys()

        # print(locdata.data)
        # print(locdata.data.dtypes)
        # print(manifest_from_locdata(locdata))
        assert_frame_equal(locdata_.data, locdata.data, check_dtype=True)

        # passing manifest
        manifest = manifest_from_locdata(locdata, return_json_string=True)
        save_SMLM(locdata, path=file_path, manifest=manifest)

        manifest = manifest_from_locdata(locdata)
        save_SMLM(locdata, path=file_path, manifest=manifest)

        # read back smlm manifest
        manifest = load_SMLM_manifest(path=file_path)
        for key in ["format_version", "formats", "files"]:
            assert key in manifest.keys()

        # read back smlm file
        locdata_ = load_SMLM_file(path=file_path)
        assert_frame_equal(locdata_.data, locdata.data, check_dtype=False)
