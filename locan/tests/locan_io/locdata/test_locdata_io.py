import logging
import pickle
import tempfile
from io import StringIO
from pathlib import Path

import pytest

import locan.constants
from locan.data import metadata_pb2
from locan.dependencies import HAS_DEPENDENCY
from locan.locan_io import load_locdata, load_txt_file
from locan.locan_io.locdata.io_locdata import _map_file_type_to_load_function


def test_loading_txt_file(caplog):
    dat = load_txt_file(
        path=locan.ROOT_DIR / "tests/test_data/five_blobs.txt", nrows=10
    )
    # print(dat.data)
    assert len(dat) == 10

    dat = load_txt_file(
        path=locan.ROOT_DIR / "tests/test_data/five_blobs.txt",
        columns=["index", "position_x", "position_y", "cluster_label", "frame"],
        nrows=10,
    )
    # print(dat.data)
    assert len(dat) == 10

    file_like = StringIO("index,position_x,position_y,cluster_label\n0,624,919,3")
    dat = load_txt_file(
        path=file_like,
        columns=["index", "position_x", "position_y", "cluster_label"],
        nrows=1,
    )
    assert len(dat) == 1

    dat = load_txt_file(
        path=locan.ROOT_DIR / "tests/test_data/five_blobs.txt", columns=["c1"], nrows=10
    )
    # print(dat.data)
    assert caplog.record_tuples == [
        (
            "locan.locan_io.locdata.utilities",
            logging.WARNING,
            "Column c1 is not a Locan property standard.",
        )
    ]

    assert len(dat) == 10

    dat = load_txt_file(
        path=locan.ROOT_DIR / "tests/test_data/five_blobs.txt",
        columns=["c1"],
        nrows=10,
        property_mapping={"c1": "something"},
    )
    assert list(dat.data.columns) == ["something"]


def test__map_file_type_to_load_function():
    file_type = _map_file_type_to_load_function(file_type=1)
    assert callable(file_type)
    file_type = _map_file_type_to_load_function(file_type="RAPIDSTORM")
    assert callable(file_type)
    file_type = _map_file_type_to_load_function(
        file_type=locan.constants.FileType.RAPIDSTORM
    )
    assert callable(file_type)
    file_type = _map_file_type_to_load_function(file_type=metadata_pb2.RAPIDSTORM)
    assert callable(file_type)


def test_load_locdata():
    dat = load_locdata(
        path=locan.ROOT_DIR / "tests/test_data/rapidSTORM_dstorm_data.txt",
        file_type="RAPIDSTORM",
        nrows=10,
    )
    assert len(dat) == 10

    dat = load_locdata(
        path=locan.ROOT_DIR / "tests/test_data/Elyra_dstorm_data.txt",
        file_type="ELYRA",
        nrows=10,
    )
    assert len(dat) == 10

    dat = load_locdata(
        path=locan.ROOT_DIR / "tests/test_data/Elyra_dstorm_data.txt",
        file_type="elyra",
        nrows=10,
    )
    assert len(dat) == 10

    dat = load_locdata(
        path=locan.ROOT_DIR / "tests/test_data/Elyra_dstorm_data.txt",
        file_type=3,
        nrows=10,
    )
    assert len(dat) == 10

    dat = load_locdata(
        path=locan.ROOT_DIR / "tests/test_data/Nanoimager_dstorm_data.csv",
        file_type="NANOIMAGER",
        nrows=10,
    )
    assert len(dat) == 10

    dat = load_locdata(
        path=locan.ROOT_DIR / "tests/test_data/rapidSTORM_dstorm_track_data.txt",
        file_type="RAPIDSTORMTRACK",
        nrows=10,
    )
    assert len(dat) == 10

    dat = load_locdata(
        path=locan.ROOT_DIR / "tests/test_data/SMLM_dstorm_data.smlm",
        file_type="SMLM",
        nrows=10,
    )
    assert len(dat) == 10


@pytest.mark.skipif(not HAS_DEPENDENCY["h5py"], reason="requires h5py.")
def test_load_locdata_2():
    dat = load_locdata(
        path=locan.ROOT_DIR / "tests/test_data/decode_dstorm_data.h5",
        file_type="DECODE",
        nrows=10,
    )
    assert len(dat) == 10

    dat = load_locdata(
        path=locan.ROOT_DIR / "tests/test_data/smap_dstorm_data.mat",
        file_type="SMAP",
        nrows=10,
    )
    assert len(dat) == 10


def test_pickling_locdata(locdata_2d):
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / "pickled_locdata.pickle"
        with open(file_path, "wb") as file:
            pickle.dump(locdata_2d, file, pickle.HIGHEST_PROTOCOL)
        with open(file_path, "rb") as file:
            locdata = pickle.load(file)  # noqa S301
        assert len(locdata_2d) == len(locdata)
        assert isinstance(locdata.meta, metadata_pb2.Metadata)
