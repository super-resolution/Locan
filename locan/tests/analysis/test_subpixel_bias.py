import matplotlib.pyplot as plt  # needed for visual inspection  # noqa: F401
import numpy as np
import pandas as pd
import pytest

from locan import LocData
from locan.analysis import SubpixelBias
from locan.analysis.subpixel_bias import _subpixel_bias


@pytest.fixture()
def locdata_simple():
    locdata_dict = {
        "position_x": [0, 0, 1, 4, 5],
        "position_y": [0, 1, 3, 4, 1],
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


def test__subpixel_bias(locdata_simple):
    results = _subpixel_bias(locdata=locdata_simple, pixel_size=2)
    assert np.array_equal(results.columns, ["position_x_modulo", "position_y_modulo"])
    assert np.array_equal(results.position_x_modulo.to_numpy(), [0, 0, 1, 0, 1])
    assert np.array_equal(results.position_y_modulo.to_numpy(), [0, 1, 1, 0, 1])


def test_SubpixelBias_empty(caplog):
    SubpixelBias(pixel_size=2).compute(LocData())
    assert caplog.record_tuples == [
        ("locan.analysis.subpixel_bias", 30, "Locdata is empty.")
    ]


def test_SubpixelBias(locdata_simple):
    sb = SubpixelBias(pixel_size=2).compute(locdata_simple)
    assert repr(sb) == "SubpixelBias(pixel_size=2)"
    assert np.array_equal(sb.results.position_x_modulo.to_numpy(), [0, 0, 1, 0, 1])
    assert np.array_equal(sb.results.position_y_modulo.to_numpy(), [0, 1, 1, 0, 1])
    sb.hist()
    # plt.show()

    plt.close("all")


@pytest.mark.visual
def test_DistributionFits(locdata_rapidSTORM_2d):
    sb = SubpixelBias(pixel_size=130).compute(locdata_rapidSTORM_2d)
    sb.hist()
    plt.legend()
    plt.show()

    plt.close("all")
