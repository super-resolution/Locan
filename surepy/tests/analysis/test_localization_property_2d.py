import pytest
import matplotlib.pyplot as plt  # needed for visual inspection

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
    assert 'model_result' in lprop.results._fields
    assert lprop.results.model_result.params
    # lprop.report()
    lprop.plot()
    lprop.plot_residuals()
    lprop.plot_deviation_from_mean()
    lprop.plot_deviation_from_median()
    # plt.show()