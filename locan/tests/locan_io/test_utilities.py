from platform import system
from io import StringIO, BytesIO, TextIOWrapper
import logging

import pytest
import numpy as np
from locan import ROOT_DIR
from locan.constants import DECODE_KEYS, THUNDERSTORM_KEYS
from locan.locan_io import convert_property_types, convert_property_names

from locan.locan_io.locdata.utilities import open_path_or_file_like


def test_convert_property_types(locdata_2d):
    df = locdata_2d.data.copy()
    types_mapping = {'position_x': 'float', 'position_y': str, 'frame': np.int64, 'not_in_there': 'float'}
    converted_df = convert_property_types(dataframe=df, loc_properties=None, types=types_mapping)
    assert all(converted_df.dtypes == [np.float32, object, np.int64, int])
    assert isinstance(converted_df['position_y'].iloc[0], str)


def test_open_path_or_file_like():
    if system() == 'Linux':
        inputs = [
            str(ROOT_DIR) + '/tests/test_data/rapidSTORM_dstorm_data.txt',
            ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
        ]
    elif system() == 'Windows':
        inputs = [
            str(ROOT_DIR) + '/tests/test_data/rapidSTORM_dstorm_data.txt',
            str(ROOT_DIR) + r'\tests\test_data\rapidSTORM_dstorm_data.txt',
            ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
            ROOT_DIR / r'tests\test_data\rapidSTORM_dstorm_data.txt'
        ]
    else:
        inputs = [
            str(ROOT_DIR) + '/tests/test_data/rapidSTORM_dstorm_data.txt',
            ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
        ]

    for pfl in inputs:
        with open_path_or_file_like(path_or_file_like=pfl) as data:
            assert isinstance(data, TextIOWrapper)
            out = data.read(10)
            assert out and isinstance(out, str)
            assert not data.closed
        assert data.closed

    pfl = StringIO("This is a file-like object to be read.")
    with open_path_or_file_like(path_or_file_like=pfl) as data:
        assert isinstance(data, StringIO)
        out = data.read(10)
        assert out and isinstance(out, str)
        assert not data.closed
    assert data.closed

    pfl = BytesIO(b"This is a file-like object to be read.")
    with open_path_or_file_like(path_or_file_like=pfl) as data:
        assert isinstance(data, BytesIO)
        out = data.read(10)
        assert out and isinstance(out, bytes)
        assert not data.closed
    assert data.closed

    pfl = ROOT_DIR / 'tests/test_data/some_file_that_does_not_exits.txt'
    with pytest.raises(FileNotFoundError):
        with open_path_or_file_like(path_or_file_like=pfl):
            pass


def test_convert_property_names(caplog):
    new_properties = convert_property_names(properties=['position_x'],
                                            property_mapping=None)
    assert new_properties == ['position_x']

    properties = ['position_x', 'y', 'z', 'frame_ix', 'xyz_cr', 'something']
    new_properties = convert_property_names(properties=properties,
                                            property_mapping=None)
    assert new_properties == ['position_x', 'y', 'z', 'frame_ix', 'x_cr', 'y_cr', 'z_cr', 'something']
    assert caplog.record_tuples[0] == ('locan.locan_io.locdata.utilities', logging.WARNING,
                                       'Column y is not a Locan property standard.')

    new_properties = convert_property_names(properties=properties,
                                            property_mapping=DECODE_KEYS)
    assert new_properties == ['position_x', 'position_y', 'position_z', 'frame', 'x_cr', 'y_cr', 'z_cr', 'something']
    assert caplog.record_tuples[4] == ('locan.locan_io.locdata.utilities', logging.WARNING,
                                       'Column something is not a Locan property standard.')

    properties = ['position_x', 'y', 'z', 'frame_ix', 'xyz_cr', 'something', 'sigma [nm]']
    new_properties = convert_property_names(properties=properties,
                                            property_mapping=[DECODE_KEYS, THUNDERSTORM_KEYS])
    assert new_properties == ['position_x', 'position_y', 'position_z', 'frame', 'x_cr', 'y_cr', 'z_cr', 'something', 'psf_sigma']
    assert caplog.record_tuples[4] == ('locan.locan_io.locdata.utilities', logging.WARNING,
                                       'Column something is not a Locan property standard.')
