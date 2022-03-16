import numpy as np
import pytest

import locan as lc
from locan.dependencies import HAS_DEPENDENCY
from locan import load_SMAP_header, load_SMAP_file


pytestmark = pytest.mark.skipif(not HAS_DEPENDENCY["h5py"], reason="requires h5py.")


def test_load_SMAP_header(caplog):
    columns = load_SMAP_header(path=lc.ROOT_DIR / 'tests/test_data/smap_dstorm_data.mat')
    assert np.array_equal(columns,
                          ['LLrel', 'PSFxnm', 'PSFxpix', 'PSFynm', 'PSFypix', 'local_background', 'channel',
                           'filenumber', 'frame', 'iterations', 'locprecnm', 'locprecznm', 'logLikelihood',
                           'intensity', 'photerr', 'position_x', 'uncertainty_x', 'xpix', 'xpixerr', 'position_y',
                           'uncertainty_y', 'ypix', 'ypixerr', 'zerr', 'position_z']
                          )
    assert caplog.record_tuples[0] == ('locan.locan_io.locdata.smap_file', 30,
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
    assert caplog.record_tuples[0] == ('locan.locan_io.locdata.smap_file', 30,
                                       'Column LLrel is not a Locan property standard.')
