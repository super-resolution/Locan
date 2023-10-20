import numpy as np
import pandas as pd
import pytest

from locan import LocData
from locan.analysis import LocalizationUncertainty
from locan.analysis.uncertainty import (
    _localization_uncertainty,
    localization_precision_model_1,
    localization_precision_model_2,
    localization_precision_model_3,
)


@pytest.fixture()
def locdata_simple():
    locdata_dict = {
        "position_x": [0, 0, 1, 4, 5],
        "position_y": [0, 1, 3, 4, 1],
        "intensity": [0, 1**2, 3**2, 4**2, 1],
        "psf_sigma_x": [100, 100, 100, 100, 100],
        "pixel_size": [10, 10, 10, 10, 10],
        "local_background": [0.1, 0.1, 0.1, 0.1, 0.1],
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


def test_localization_precision_model_1():
    intensities = [1**2, 2**2, 3**2]
    uncertainties = localization_precision_model_1(intensity=intensities)
    assert isinstance(uncertainties, np.ndarray)
    assert np.array_equal(uncertainties, [1, 2, 3])


def test_localization_precision_model_2():
    intensities = [1**2, 2**2, 3**2]
    uncertainties = localization_precision_model_2(intensity=intensities, psf_sigma=12)
    assert isinstance(uncertainties, np.ndarray)
    assert np.array_equal(uncertainties, [12, 6.0, 4])

    sigma_psf_list = [12, 12, 12]
    uncertainties = localization_precision_model_2(
        intensity=intensities, psf_sigma=sigma_psf_list
    )
    assert isinstance(uncertainties, np.ndarray)
    assert np.array_equal(uncertainties, [12, 6.0, 4])


def test_localization_precision_model_3():
    intensities = [1**2, 2**2, 3**2]
    uncertainties = localization_precision_model_3(
        intensity=intensities, psf_sigma=12, pixel_size=10, local_background=0.1
    )
    assert isinstance(uncertainties, np.ndarray)
    np.testing.assert_allclose(uncertainties, [28.8, 9.6, 5.5], rtol=1e-1)

    sigma_psf_list = [12, 12, 12]
    background_list = [0.1, 0.1, 0.1]
    uncertainties = localization_precision_model_3(
        intensity=intensities,
        psf_sigma=sigma_psf_list,
        pixel_size=10,
        local_background=background_list,
    )
    assert isinstance(uncertainties, np.ndarray)
    np.testing.assert_allclose(uncertainties, [28.8, 9.6, 5.5], rtol=1e-1)

    uncertainties = localization_precision_model_3(
        intensity=2000,
        psf_sigma=150,
        pixel_size=130,
        local_background=100,
    )
    assert uncertainties == pytest.approx(6.32, rel=0.01)


def test__localization_uncertainty(locdata_simple, caplog):
    results = _localization_uncertainty(locdata=locdata_simple, model=1)
    assert caplog.record_tuples == [
        (
            "locan.analysis.uncertainty",
            30,
            "The localization property `intensity` does not have the unit photons.",
        )
    ]
    assert all(key_ in results.columns for key_ in ["uncertainty_x", "uncertainty_y"])

    with pytest.warns(RuntimeWarning):
        results = _localization_uncertainty(locdata=locdata_simple, model=2)
    assert all(key_ in results.columns for key_ in ["uncertainty_x"])
    assert results.iloc[-1, 0] == pytest.approx(100, rel=0.01)

    with pytest.warns(RuntimeWarning):
        results = _localization_uncertainty(
            locdata=locdata_simple, model=2, psf_sigma_x=150
        )
    assert all(key_ in results.columns for key_ in ["uncertainty_x"])
    assert results.iloc[-1, 0] == pytest.approx(150, rel=0.01)

    with pytest.warns(RuntimeWarning):
        results = _localization_uncertainty(locdata=locdata_simple, model=3)
    assert all(key_ in results.columns for key_ in ["uncertainty_x"])
    assert results.iloc[-1, 0] == pytest.approx(1592.0, rel=0.01)

    with pytest.warns(RuntimeWarning):
        results = _localization_uncertainty(
            locdata=locdata_simple, model=3, pixel_size=130
        )
    assert all(key_ in results.columns for key_ in ["uncertainty_x"])
    assert results.iloc[-1, 0] == pytest.approx(192.8, rel=0.01)

    def my_callable(intensity, factor):
        intensity = np.asarray(intensity)
        return intensity * factor

    results = _localization_uncertainty(
        locdata=locdata_simple, model=my_callable, factor=2
    )
    assert all(key_ in results.columns for key_ in ["uncertainty_x", "uncertainty_y"])
    assert results.iloc[-1, 0] == pytest.approx(2, rel=0.01)

    with pytest.raises(KeyError):
        _localization_uncertainty(
            locdata=locdata_simple, model=my_callable, not_existing=2
        )

    results = _localization_uncertainty(locdata=LocData(), model=1)
    assert results.empty


class TestLocalizationUncertainty:
    def test_empty_locdata(self, caplog):
        LocalizationUncertainty().compute(locdata=LocData())
        assert caplog.record_tuples == [
            ("locan.analysis.uncertainty", 30, "Locdata is empty.")
        ]

    def test_init(self, locdata_simple):
        with pytest.warns(RuntimeWarning):
            unc = LocalizationUncertainty(model=2).compute(locdata_simple)
        assert all(key_ in unc.results.columns for key_ in ["uncertainty_x"])
        assert unc.results.iloc[-1, 0] == pytest.approx(100, rel=0.01)

    def test_kwargs(self, locdata_simple):
        with pytest.warns(RuntimeWarning):
            unc = LocalizationUncertainty(model=2, psf_sigma_x=130).compute(
                locdata_simple
            )
        assert all(key_ in unc.results.columns for key_ in ["uncertainty_x"])
        assert unc.results.iloc[-1, 0] == pytest.approx(130, rel=0.01)

        with pytest.warns(RuntimeWarning):
            unc = LocalizationUncertainty(
                model=localization_precision_model_3, pixel_size=130
            ).compute(locdata_simple)
        assert all(key_ in unc.results.columns for key_ in ["uncertainty_x"])
        assert unc.results.iloc[-1, 0] == pytest.approx(192.8, rel=0.01)
