"""
Compute localization uncertainty.

A theoretical estimate for localization uncertainty is given by the
Cramer-Rao-lower-bound for the localization precision.
The localization precision depends on a number of experimental factors
including camera and photophysical characteristics [1]_ [2]_.
We provide functions to compute localization uncertainty from available
localization properties as predicted by the Cramer-Rao-lower-bound for the
localization precision.

References
----------
.. [1] K.I. Mortensen, L. S. Churchman, J. A. Spudich, H. Flyvbjerg,
   Nat. Methods 7 (2010): 377–384.
.. [2] Rieger B., Stallinga S.,
   The lateral and axial localization uncertainty in super-resolution
   light microscopy.
   Chemphyschem 17;15(4), 2014:664-70. doi: 10.1002/cphc.201300711
"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable
from inspect import signature
from typing import Any, Protocol

import numpy as np
import pandas as pd

from locan.analysis.analysis_base import _Analysis

__all__ = ["LocalizationUncertainty", "LocalizationUncertaintyFromIntensity"]

logger = logging.getLogger(__name__)


# The algorithms

LocalizationPrecisionModel = Callable[[...], np.ndarray]


class Locdata(Protocol):
    data: Any
    meta: Any
    coordinate_labels: Iterable


def localization_precision_model_1(intensity) -> np.ndarray:
    """
    Localization precision as function of
    intensity (I) according to:

    .. math::

        sigma_{loc} ) = \frac{1}{\\sqrt{I}}


    Parameters
    ----------
    intensity : array-like
        Intensity values

    Returns
    -------
    numpy.ndarray
    """
    intensity = np.asarray(intensity)
    sigma = np.sqrt(intensity)
    return sigma


def localization_precision_model_2(intensity, psf_sigma) -> np.ndarray:
    """
    Localization precision as function of
    intensity (I), PSF width (sigma_PSF) according to:

    .. math::

        sigma_{loc} ) = \frac{sigma_{PSF}}{\\sqrt{I}}

    Parameters
    ----------
    intensity : array-like
        Intensity values
    psf_sigma : array-like
        The PSF size

    Returns
    -------
    numpy.ndarray
    """
    intensity = np.asarray(intensity)
    psf_sigma = np.asarray(psf_sigma)
    sigma = psf_sigma / np.sqrt(intensity)
    return sigma


def localization_precision_model_3(
    intensity, psf_sigma, pixel_size, local_background
) -> np.ndarray:
    """
    Localization precision as function of
    intensity (I), PSF width (sigma_PSF), pixel size (), background
    according to:

    .. math::

        sigma_{loc} ) = \frac{sigma_{PSF}}{\\sqrt{I}}

    Parameters
    ----------
    intensity : array-like
        Intensity values
    psf_sigma : array-like
        The PSF size
    pixel_size : array-like
        Size of camera pixel
    local_background : array-like
        The local background

    Returns
    -------
    numpy.ndarray
    """
    intensity = np.asarray(intensity)
    psf_sigma = np.asarray(psf_sigma)
    pixel_size = np.asarray(pixel_size)
    local_background = np.asarray(local_background)

    sigma_a_squared = psf_sigma**2 + pixel_size**2 / 12
    tau = 2 * np.pi * sigma_a_squared * local_background / (intensity * pixel_size**2)

    sigma_squared = (
        sigma_a_squared / intensity * (1 + 4 * tau + np.sqrt(2 * tau / (1 + 4 * tau)))
    )
    sigma = np.sqrt(sigma_squared)
    return sigma


def _localization_uncertainty(
    locdata: Locdata, model: int | LocalizationPrecisionModel, **kwargs: dict
):
    if "intensity" not in locdata.data.columns:
        raise KeyError("Localization property `intensity` is not available.")

    # todo: improve checks on localization_property units
    try:
        if (
            not [
                prop_.unit
                for prop_ in locdata.meta.localization_properties
                if prop_.name == "intensity"
            ][0]
            == "photons"
        ):
            raise UserWarning
    except (UserWarning, IndexError):
        logger.warning(
            "The localization property `intensity` does not have the unit photons."
        )

    # todo: check if unit psf_sigma and coordinates is the same.

    if isinstance(model, Callable):
        pass
    elif model == 1:
        model = localization_precision_model_1
    elif model == 2:
        model = localization_precision_model_2
    elif model == 3:
        model = localization_precision_model_3
    else:
        raise TypeError("model must be 1, 2, 3 or callable.")

    label_suffixes = [""]
    label_suffixes.extend([label_[-2:] for label_ in locdata.coordinate_labels])

    params = signature(model).parameters.keys()
    temp_dict = {}
    for suffix_ in label_suffixes:
        temp_dict[suffix_] = [
            p_ + suffix_
            if p_ + suffix_ in locdata.data.columns
            else (p_ if p_ in locdata.data.columns else None)
            for p_ in params
        ]
    params_dict = {} if None in temp_dict[""] else {"": temp_dict[""]}
    for key, values in temp_dict.items():
        if None not in values and temp_dict[""] != values:
            params_dict[key] = values

    results_dict = {}
    for key, value in params_dict.items():
        results_key = "uncertainty" + key
        args_ = [locdata.data[item_].to_numpy() for item_ in value]

        for kwarg_key, kwarg_value in kwargs.items():
            if kwarg_key in params:
                index = list(params).index(kwarg_key)
                args_[index] = kwarg_value

        results_dict[results_key] = model(*args_)

    return pd.DataFrame(results_dict)


def _localization_uncertainty_from_intensity(locdata):
    results = {}
    for v in ["x", "y", "z"]:
        if (
            "position_" + v in locdata.data.keys()
            and "intensity" in locdata.data.keys()
        ):
            if "psf_sigma_" + v in locdata.data.keys():
                results.update(
                    {
                        "uncertainty_"
                        + v: locdata.data["psf_sigma_" + v]
                        / np.sqrt(locdata.data["intensity"])
                    }
                )
            else:
                results.update(
                    {"uncertainty_" + v: 1 / np.sqrt(locdata.data["intensity"])}
                )
        else:
            pass

    return pd.DataFrame(results)


# The specific analysis classes


class LocalizationUncertaintyFromIntensity(_Analysis):
    """
    Compute the localization uncertainty for each localization's spatial
    coordinate in locdata.

    Uncertainty is computed as Psf_sigma / Sqrt(Intensity) for each spatial dimension.
    If Psf_sigma is not available Uncertainty is 1 / Sqrt(Intensity).

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.DataFrame
        The number of localizations per frame or
        the number of localizations per frame normalized to region_measure(hull).
    """

    count = 0

    def __init__(self, meta=None):
        super().__init__(meta=meta)
        self.results = None
        warnings.warn(
            f"{self.__class__.__name__} will be deprecated. "
            f"Use `LocalizationUncertainty` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def compute(self, locdata):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
            Localization data.

        Returns
        -------
        Analysis class
            Returns the Analysis class object (self).
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        self.results = _localization_uncertainty_from_intensity(locdata=locdata)
        return self


class LocalizationUncertainty(_Analysis):
    """
    Compute the Cramer Rao lower bound for localization uncertainty
    for each localization's spatial coordinate in locdata.

    Uncertainty is computed according to one of the following model functions:

    1) 1 / sqrt(intensity)
    2) psf_sigma / sqrt(intensity)
    3) f(psf_sigma, intensity, pixel_size, background)

    Localization properties have to be available for all or each spatial
    dimension (like `psf_sigma` or `psf_sigma_x`).
    The localization property `intensity` must have the unit `photons`.
    The unit of `pixel_size` must be the same as that of position coordinates.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    model : int
        Model function for theoretical localization uncertainty.
    kwargs : dict
        kwargs for the chosen model.
        If none are given the localization properties are taken from locdata.


    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.DataFrame
        Uncertainty for each localization and in each dimension.
    """

    count = 0

    def __init__(self, meta=None, model=1, **kwargs):
        super().__init__(meta=meta, model=model, **kwargs)
        self.results = None

    def compute(self, locdata):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
            Localization data.

        Returns
        -------
        Analysis class
            Returns the Analysis class object (self).
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        self.results = _localization_uncertainty(locdata=locdata, **self.parameter)
        return self
