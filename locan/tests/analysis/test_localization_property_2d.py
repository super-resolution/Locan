from copy import deepcopy

import matplotlib.pyplot as plt  # needed for visual inspection  # noqa: F401
import pytest

from locan import (
    LocalizationProperty2d,
    LocData,
    Rectangle,
    render_2d_mpl,  # needed for visual inspection  # noqa: F401
    simulate_uniform,
)
from locan.analysis.localization_property_2d import _gauss_2d


def test_Localization_property_2d_empty(caplog):
    lprop = LocalizationProperty2d().compute(LocData())
    lprop.plot()
    lprop.plot_residuals()
    lprop.plot_deviation_from_mean()
    lprop.plot_deviation_from_median()
    lprop.report()
    assert caplog.record_tuples == [
        ("locan.analysis.localization_property_2d", 30, "Locdata is empty."),
        ("locan.analysis.localization_property_2d", 30, "No results available"),
    ]


@pytest.mark.visual
def test_Localization_property_2d_simulated_data(capfd):
    locdata = simulate_uniform(n_samples=10_000, region=Rectangle((0, 500), 1000, 500))
    intensity = _gauss_2d(
        x=locdata.data.position_x,
        y=locdata.data.position_y,
        amplitude=1000,
        center_x=800,
        center_y=700,
        sigma_x=300,
        sigma_y=300,
    )
    df = locdata.dataframe.assign(intensity=intensity)
    locdata = locdata.update(dataframe=df)

    # render_2d_mpl(locdata, other_property="intensity", bin_size=20)

    lprop = LocalizationProperty2d(
        meta=None, other_property="intensity", bin_size=(20, 50)
    ).compute(locdata)
    assert "model_result" in lprop.results._fields
    assert lprop.results.model_result.params

    lprop.report()
    captured = capfd.readouterr()
    assert captured.out[:16] == "Fit results for:"

    lprop.plot()
    lprop.plot_residuals()
    lprop.plot_deviation_from_mean()
    lprop.plot_deviation_from_median()
    plt.show()

    plt.close("all")


def test_Localization_property_2d(capfd, locdata_blobs_2d):
    locdata = deepcopy(locdata_blobs_2d)
    intensity = _gauss_2d(
        x=locdata.data.position_x,
        y=locdata.data.position_y,
        amplitude=1000,
        center_x=800,
        center_y=700,
        sigma_x=300,
        sigma_y=300,
    )
    df = locdata.dataframe.assign(intensity=intensity)
    locdata = locdata.update(dataframe=df)

    lprop = LocalizationProperty2d(
        meta=None, other_property="intensity", bin_size=20
    ).compute(locdata)
    assert "model_result" in lprop.results._fields
    assert lprop.results.model_result.params

    lprop.report()
    captured = capfd.readouterr()
    assert captured.out[:16] == "Fit results for:"

    lprop.plot(colors="r")
    lprop.plot_residuals()
    lprop.plot_deviation_from_mean()
    lprop.plot_deviation_from_median()
    # plt.show()

    plt.close("all")
