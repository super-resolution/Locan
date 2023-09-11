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
import sys
import warnings
from collections.abc import Callable
from inspect import signature
from typing import TYPE_CHECKING, cast

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from locan.data.locdata import LocData

import numpy as np
import numpy.typing as npt  # noqa: F401
import pandas as pd

from locan.analysis.analysis_base import _Analysis
from locan.data.locdata_utils import _get_loc_property_key_per_dimension

__all__: list[str] = ["LocalizationUncertainty", "LocalizationUncertaintyFromIntensity"]

logger = logging.getLogger(__name__)


# The algorithms

if sys.version_info < (3, 9):
    LocalizationPrecisionModel = Callable
else:
    LocalizationPrecisionModel = Callable[..., npt.NDArray[np.float_]]


def localization_precision_model_1(intensity) -> npt.NDArray[np.float_]:
    """
    Localization precision as function of
    intensity (I) according to:

    .. math::

        \\sigma_{loc} = \\frac{1}{\\sqrt{I}}


    Parameters
    ----------
    intensity : npt.ArrayLike
        Intensity values

    Returns
    -------
    npt.NDArray[np.float_]
    """
    intensity = np.asarray(intensity)
    sigma = np.sqrt(intensity)
    return sigma


def localization_precision_model_2(intensity, psf_sigma) -> npt.NDArray[np.float_]:
    """
    Localization precision as function of
    intensity (I), PSF width (sigma_PSF) according to:

    .. math::

        \\sigma_{loc} = \\frac{\\sigma_{PSF}}{\\sqrt{I}}

    Parameters
    ----------
    intensity : npt.ArrayLike
        Intensity values
    psf_sigma : npt.ArrayLike
        The PSF size

    Returns
    -------
    npt.NDArray[np.float_]
    """
    intensity = np.asarray(intensity)
    psf_sigma = np.asarray(psf_sigma)
    sigma = psf_sigma / np.sqrt(intensity)
    return sigma


def localization_precision_model_3(
    intensity, psf_sigma, pixel_size, local_background
) -> npt.NDArray[np.float_]:
    """
    Localization precision as function of
    intensity (I), PSF size (\\sigma_{PSF}), pixel size (a), background (b)
    according to:

    .. math::

        \\sigma_{a} ^{2} = \\sigma_{PSF} ^{2} + a^{2} / 12

        \\tau = 2 * \\pi * b * \\sigma_{a} ^{2} / (I * a^{2})

        \\sigma_{loc} ^{2} = \\sigma_{a} ^{2} / I * (1 + 4 * \\tau + \\sqrt{2 * \\tau / (1 + 4 * \\tau) } )


    Parameters
    ----------
    intensity : npt.ArrayLike
        Intensity values
    psf_sigma : npt.ArrayLike
        The PSF size
    pixel_size : npt.ArrayLike
        Size of camera pixel
    local_background : npt.ArrayLike
        The local background

    Returns
    -------
    npt.NDArray[np.float_]
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
    locdata: LocData, model: int | LocalizationPrecisionModel, **kwargs
) -> pd.DataFrame:
    if isinstance(model, Callable):  # type: ignore
        model = cast(Callable, model)  # type: ignore[type-arg]
    elif model == 1:
        model = cast(Callable, model)  # type: ignore[type-arg]
        model = localization_precision_model_1
    elif model == 2:
        model = cast(Callable, model)  # type: ignore[type-arg]
        model = localization_precision_model_2
    elif model == 3:
        model = cast(Callable, model)  # type: ignore[type-arg]
        model = localization_precision_model_3
    else:
        raise TypeError("model must be 1, 2, 3 or callable.")

    params = signature(model).parameters.keys()

    for key_ in kwargs.keys():
        if key_ not in params:
            if key_[:-2] not in params:
                raise KeyError(f"Kwarg {key_} does not fit the models signature.")

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

    available_keys_list = [
        _get_loc_property_key_per_dimension(locdata=locdata.data, property_key=param_)
        for param_ in params
    ]
    available_coordinate_keys = _get_loc_property_key_per_dimension(
        locdata=locdata.data, property_key="position"
    )
    new_uncertainty_keys = ["uncertainty_x", "uncertainty_y", "uncertainty_z"]

    results_dict = dict()
    for i, (c_key, new_u_key) in enumerate(
        zip(available_coordinate_keys, new_uncertainty_keys)
    ):
        # transpose available_keys_list
        available_keys = [item[i] for item in available_keys_list]
        if c_key is not None:
            suffix = "_" + c_key.split("_")[-1]
            args = []
            # go through all args for model
            for key_, param_ in zip(available_keys, params):  # type: ignore
                if key_ in kwargs.keys():
                    args.append(kwargs[key_])
                elif key_ is None and param_ in kwargs.keys():
                    args.append(kwargs[param_])
                elif key_ is None and param_ + suffix in kwargs.keys():
                    args.append(kwargs[param_ + suffix])
                elif key_ is None:
                    args.append(None)
                    break
                else:
                    args.append(locdata.data[key_])
            if all(arg_ is not None for arg_ in args):
                results_dict[new_u_key] = model(*args)

    return pd.DataFrame(results_dict)


def _localization_uncertainty_from_intensity(locdata: LocData) -> pd.DataFrame:
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

    .. deprecated:: 0.14
    LocalizationUncertaintyFromIntensity is deprecated.
    Use `LocalizationUncertainty` instead.

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

    def __init__(self, meta=None) -> None:
        super().__init__(meta=meta)
        self.results = None
        warnings.warn(
            f"{self.__class__.__name__} will be deprecated. "
            f"Use `LocalizationUncertainty` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def compute(self, locdata: LocData) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
            Localization data.

        Returns
        -------
        Self
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
    meta : locan.analysis.metadata_analysis_pb2.AMetadata | None
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

    def __init__(self, meta=None, model=1, **kwargs) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.results = None

    def compute(self, locdata: LocData) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
            Localization data.

        Returns
        -------
        Self
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        self.results = _localization_uncertainty(locdata=locdata, **self.parameter)
        return self
