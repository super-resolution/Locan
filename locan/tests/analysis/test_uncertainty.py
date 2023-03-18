import numpy as np
import pandas as pd
import pytest

from locan import LocData
from locan.analysis import LocalizationUncertainty, LocalizationUncertaintyFromIntensity
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


def test__localization_uncertainty(locdata_simple, caplog):
    results = _localization_uncertainty(locdata=locdata_simple, model=1)
    # print(results)
    assert caplog.record_tuples == [
        (
            "locan.analysis.uncertainty",
            30,
            "The localization property `intensity` does not have the unit photons.",
        )
    ]
    assert all(key_ in results.columns for key_ in ["uncertainty"])

    results = _localization_uncertainty(locdata=locdata_simple, model=2)
    # print(results)
    assert all(key_ in results.columns for key_ in ["uncertainty_x"])

    results = _localization_uncertainty(locdata=locdata_simple, model=3)
    assert all(key_ in results.columns for key_ in ["uncertainty_x"])

    def my_callable():
        return np.array([])

    results = _localization_uncertainty(locdata=locdata_simple, model=my_callable)
    assert results.empty

    with pytest.raises(KeyError):
        _localization_uncertainty(locdata=LocData(), model=1)


def test_LocalizationUncertainty_empty(caplog):
    LocalizationUncertainty().compute(LocData())
    assert caplog.record_tuples == [
        ("locan.analysis.uncertainty", 30, "Locdata is empty.")
    ]


def test_uncertainty(locdata_simple):
    unc = LocalizationUncertainty(model=3).compute(locdata_simple)
    assert all(key_ in unc.results.columns for key_ in ["uncertainty_x"])


# todo: deprecate in v0.15
def test_uncertainty_empty_to_be_deprecated(caplog):
    LocalizationUncertaintyFromIntensity().compute(LocData())
    assert caplog.record_tuples == [
        ("locan.analysis.uncertainty", 30, "Locdata is empty.")
    ]


# todo: deprecate in v0.15
def test_uncertainty_to_be_deprecated(locdata_simple):
    unc = LocalizationUncertaintyFromIntensity().compute(locdata_simple)
    # print(unc.results)
    # print(unc.results['Uncertainty_x'][0])
    assert unc.results["uncertainty_x"][0] == np.inf
    assert unc.results["uncertainty_x"][1] == 100
