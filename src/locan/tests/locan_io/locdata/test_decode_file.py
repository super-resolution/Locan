import numpy as np
import pytest

import locan as lc
from locan import FileType, load_decode_file, load_decode_header
from locan.dependencies import HAS_DEPENDENCY

pytestmark = pytest.mark.skipif(not HAS_DEPENDENCY["h5py"], reason="requires h5py.")


def test_load_DECODE_header():
    columns, meta, decode = load_decode_header(
        path=lc.ROOT_DIR / "tests/test_data/decode_dstorm_data.h5"
    )
    # print(columns, meta, decode)
    assert list(meta.keys()) == ["px_size", "xy_unit"]
    assert list(decode.keys()) == ["version"]
    assert columns == [
        "local_background",
        "bg_cr",
        "bg_sig",
        "frame",
        "original_index",
        "intensity",
        "phot_cr",
        "phot_sig",
        "prob",
        "position_x",
        "position_y",
        "position_z",
        "x_cr",
        "y_cr",
        "z_cr",
        "x_sig",
        "y_sig",
        "z_sig",
    ]


def test_loading_DECODE_file_empty_file():
    locdata = load_decode_file(
        path=lc.ROOT_DIR / "tests/test_data/decode_dstorm_data_empty.h5", nrows=10
    )
    assert len(locdata) == 0


def test_loading_DECODE_file():
    file_path = lc.ROOT_DIR / "tests/test_data/decode_dstorm_data.h5"
    locdata = load_decode_file(path=file_path, nrows=10)
    # print(locdata.data)
    assert len(locdata) == 10
    assert np.array_equal(
        locdata.data.columns,
        [
            "local_background",
            "frame",
            "original_index",
            "intensity",
            "phot_sig",
            "prob",
            "position_x",
            "position_y",
            "position_z",
            "x_sig",
            "y_sig",
            "z_sig",
        ],
    )
    assert locdata.meta.file.type == FileType.DECODE.value
    assert locdata.meta.file.path == str(file_path)
