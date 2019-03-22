import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.analysis import LocalizationUncertaintyFromIntensity


@pytest.fixture()
def locdata_simple():
    dict = {
        'position_x': [0, 0, 1, 4, 5],
        'position_y': [0, 1, 3, 4, 1],
        'intensity': [0, 1, 3, 4, 1],
        'psf_sigma_x': [100, 100, 100, 100, 100],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


def test_uncertainty(locdata_simple):
    unc = LocalizationUncertaintyFromIntensity(locdata_simple).compute()
    # print(unc.results)
    # print(unc.results['Uncertainty_x'][0])
    assert(unc.results['uncertainty_x'][0] == np.inf)
    assert(unc.results['uncertainty_x'][1] == 100)
