import numpy as np
import pandas as pd
import pytest

from locan import LocData
from locan.analysis import LocalizationUncertaintyFromIntensity


@pytest.fixture()
def locdata_simple():
    locdata_dict = {
        "position_x": [0, 0, 1, 4, 5],
        "position_y": [0, 1, 3, 4, 1],
        "intensity": [0, 1, 3, 4, 1],
        "psf_sigma_x": [100, 100, 100, 100, 100],
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


def test_uncertainty_empty(caplog):
    LocalizationUncertaintyFromIntensity().compute(LocData())
    assert caplog.record_tuples == [
        ("locan.analysis.uncertainty", 30, "Locdata is empty.")
    ]


def test_uncertainty(locdata_simple):
    unc = LocalizationUncertaintyFromIntensity().compute(locdata_simple)
    # print(unc.results)
    # print(unc.results['Uncertainty_x'][0])
    assert unc.results["uncertainty_x"][0] == np.inf
    assert unc.results["uncertainty_x"][1] == 100
