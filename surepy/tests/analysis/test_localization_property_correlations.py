import pytest
import pandas as pd
import matplotlib.pyplot as plt  # needed for visual inspection

from surepy import ROOT_DIR, load_rapidSTORM_file
from surepy.analysis import LocalizationPropertyCorrelations


@pytest.fixture()
def locdata():
    return load_rapidSTORM_file(path=ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt',
                                   nrows=100)


def test_Localization_property_correlations(locdata):
    lpcorr = LocalizationPropertyCorrelations().compute(locdata=locdata)
    # print(lpcorr.results)
    for i in range(len(locdata.data.columns)):
        assert lpcorr.results.iloc[i, i] == 1

    lpcorr = LocalizationPropertyCorrelations(loc_properties=['intensity', 'local_background']).compute(locdata=locdata)
    for i in range(2):
        assert lpcorr.results.iloc[i, i] == 1

    #lpcorr.plot()
    lpcorr.plot(cbar=False, vmin=0)
    # plt.show()
