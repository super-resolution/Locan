from pathlib import Path
import tempfile
from copy import deepcopy

import locan.constants
import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from google.protobuf import json_format

import locan as lc
from locan.dependencies import HAS_DEPENDENCY
from locan import load_SMAP_header, load_SMAP_file, save_SMAP_csv, load_txt_file


pytestmark = pytest.mark.skipif(not HAS_DEPENDENCY["h5py"], reason="requires h5py.")


def test_load_SMAP_header(caplog):
    columns = load_SMAP_header(path=lc.ROOT_DIR / 'tests/test_data/smap_dstorm_data.mat')
    assert np.array_equal(columns,
                          ['LLrel', 'PSFxnm', 'PSFxpix', 'PSFynm', 'PSFypix', 'local_background', 'channel',
                           'filenumber', 'frame', 'iterations', 'locprecnm', 'locprecznm', 'logLikelihood',
                           'intensity', 'photerr', 'position_x', 'uncertainty_x', 'xpix', 'xpixerr', 'position_y',
                           'uncertainty_y', 'ypix', 'ypixerr', 'zerr', 'position_z']
                          )
    assert caplog.record_tuples[0] == ('locan.locan_io.locdata.utilities', 30,
                                       'Column LLrel is not a Locan property standard.')


# this is inactive because an empty .mat file is too large to be kept as test_data
# def test_loading_SMAP_file_empty_file():
#     locdata = load_SMAP_file(
#         path=lc.ROOT_DIR / 'tests/test_data/smap_dstorm_data_empty.mat',
#         nrows=10)
#     assert len(locdata) == 0


def test_loading_SMAP_file(caplog):
    locdata = load_SMAP_file(
        path=lc.ROOT_DIR / 'tests/test_data/smap_dstorm_data.mat',
        nrows=10)
    assert len(locdata) == 10
    assert np.array_equal(locdata.data.columns,
                          ['LLrel', 'PSFxnm', 'PSFxpix', 'PSFynm', 'PSFypix', 'local_background', 'channel',
                           'filenumber', 'frame', 'iterations', 'locprecnm', 'locprecznm', 'logLikelihood',
                           'intensity', 'photerr', 'position_x', 'uncertainty_x', 'xpix', 'xpixerr', 'position_y',
                           'uncertainty_y', 'ypix', 'ypixerr', 'zerr', 'position_z']
                          )
    assert caplog.record_tuples[0] == ('locan.locan_io.locdata.utilities', 30,
                                       'Column LLrel is not a Locan property standard.')


def test_save_and_load_smap_csv(locdata_2d):
    locdata = deepcopy(locdata_2d)
    # introduce different dtypes in data
    locdata.data["position_x"] = locdata.data["position_x"].astype('float64')
    locdata.data["position_y"] = locdata.data["position_y"].astype('int32')
    locdata.data["frame"] = locdata.data["frame"].astype('int32')
    locdata.data["intensity"] = locdata.data["intensity"].astype('int32')
    assert all(locdata.data.dtypes.values == ["float64", "int32", "int32", "int32"])
    # print(locdata.data)

    with tempfile.TemporaryDirectory() as tmp_directory:
        # save smap file
        file_path = Path(tmp_directory) / 'locdata.csv'
        save_SMAP_csv(locdata, path=file_path)

        # read back smap file
        locdata_ = load_txt_file(path=file_path, property_mapping=locan.constants.SMAP_KEYS)
        assert len(locdata_) == len(locdata)
        assert (locdata_.properties.keys() == locdata.properties.keys())

        # print(locdata_.data)
        # print(locdata_.data.dtypes)
        assert_frame_equal(locdata_.data, locdata.data, check_dtype=False)
        # todo: add convert kwarg to load_txt_file
        # assert_frame_equal(locdata_.data, locdata.data, check_dtype=True)
