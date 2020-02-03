import pytest
import matplotlib.pyplot as plt

import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.analysis import LocalizationProperty2d


@pytest.fixture()
def locdata():
    return io.load_rapidSTORM_file(path=surepy.constants.ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
                                   nrows=100)


def test_Localization_property_2d(locdata):
    lprop = LocalizationProperty2d(meta=None, other_property='local_background', bin_size=1000).compute(locdata)
    assert 'fit_results' in lprop.results._fields
    # lprop.report()
    lprop.plot()
    # plt.show()