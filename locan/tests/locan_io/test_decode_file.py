import numpy as np

import locan
from locan import load_decode_header, load_decode_file
#from locan.locan_io.locdata.decode_file import load_decode_header, load_decode_file


def test_load_DECODE_header():
    columns, meta, decode = load_decode_header(path=locan.ROOT_DIR / 'tests/test_data/decode_dstorm_data.h5')
    # print(columns, meta, decode)
    assert list(meta.keys()) == ['px_size', 'xy_unit']
    assert list(decode.keys()) == ['version']
    assert columns == [
        'local_background',
        'bg_cr',
        'bg_sig',
        'frame',
        'original_index',
        'intensity',
        'phot_cr',
        'phot_sig',
        'prob',
        'position_x',
        'position_y',
        'position_z',
        'x_cr',
        'y_cr',
        'z_cr',
        'x_sig',
        'y_sig',
        'z_sig'
    ]


def test_loading_DECODE_file_empty_file():
    locdata = load_decode_file(
        path=locan.ROOT_DIR / 'tests/test_data/decode_dstorm_data_empty.h5',
        nrows=10)
    assert len(locdata) == 0


def test_loading_DECODE_file():
    locdata = load_decode_file(
        path=locan.ROOT_DIR / 'tests/test_data/decode_dstorm_data.h5',
        nrows=10)
    # print(locdata.data)
    assert len(locdata) == 10
    assert np.array_equal(locdata.data.columns, [
        'local_background', 'frame', 'original_index', 'intensity', 'phot_sig',
        'prob', 'position_x', 'position_y', 'position_z', 'x_sig', 'y_sig',
        'z_sig'
    ])
