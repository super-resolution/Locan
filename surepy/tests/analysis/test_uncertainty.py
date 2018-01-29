import pytest
import numpy as np
import pandas as pd
from surepy import LocData
from surepy.analysis import Localization_uncertainty_from_intensity


@pytest.fixture()
def locdata_simple():
    dict = {
        'Position_x': [0, 0, 1, 4, 5],
        'Position_y': [0, 1, 3, 4, 1],
        'Intensity': [0, 1, 3, 4, 1],
        'Psf_sigma_x': [100, 100, 100, 100, 100],
        }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))


def test_uncertainty(locdata_simple):
    unc = Localization_uncertainty_from_intensity(locdata_simple)
    # print(unc.results)
    # print(unc.results['Uncertainty_x'][0])
    assert(unc.results['Uncertainty_x'][0] == np.inf)
    assert(unc.results['Uncertainty_x'][1] == 100)
