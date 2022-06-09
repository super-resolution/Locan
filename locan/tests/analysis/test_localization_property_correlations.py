import matplotlib.pyplot as plt  # needed for visual inspection
import pytest

from locan import LocData
from locan.analysis import LocalizationPropertyCorrelations


def test_Localization_property_correlations_empty(caplog):
    lpcorr = LocalizationPropertyCorrelations().compute(LocData())
    lpcorr.plot()
    assert caplog.record_tuples == [
        ("locan.analysis.localization_property_correlations", 30, "Locdata is empty.")
    ]


def test_Localization_property_correlations(locdata_rapidSTORM_2d):
    lpcorr = LocalizationPropertyCorrelations().compute(locdata=locdata_rapidSTORM_2d)
    # print(lpcorr.results)
    for i in range(len(locdata_rapidSTORM_2d.data.columns)):
        assert lpcorr.results.iloc[i, i] == pytest.approx(1)

    lpcorr = LocalizationPropertyCorrelations(
        loc_properties=["intensity", "local_background"]
    ).compute(locdata=locdata_rapidSTORM_2d)
    for i in range(2):
        assert lpcorr.results.iloc[i, i] == pytest.approx(1)

    lpcorr.plot()
    # lpcorr.plot(cbar=False, vmin=0)
    # plt.show()

    plt.close("all")
