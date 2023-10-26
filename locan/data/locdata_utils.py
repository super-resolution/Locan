"""

Utility functions for working with locdata.

"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from locan.utils.statistics import weighted_mean_variance

if TYPE_CHECKING:
    from locan.data.locdata import LocData  # noqa F401
    from locan.locan_types import DataFrame  # noqa F401

__all__: list[str] = []

logger = logging.getLogger(__name__)


def _get_loc_property_key_per_dimension(
    locdata: pd.DataFrame | pd.Series[Any], property_key: str
) -> list[str | None]:
    """
    Get tuple with property_key as available in each dimension.
    For the x-coordinates precedence is given as:
    `position_x` or `position` or None.
    None if the key is not available for a certain dimension.

    Parameters
    ----------
    locdata
        Localization data
    property_key
        Property key to look for in locdata.

    Returns
    -------
    list[str | None]
        The available property keys in each dimension
    """
    available_keys: list[str | None] = []
    for extension_ in ["_x", "_y", "_z"]:
        prop_key = property_key + extension_
        if prop_key in locdata.columns:
            available_keys.append(prop_key)
        elif property_key in locdata.columns:
            available_keys.append(property_key)
        else:
            available_keys.append(None)
    return available_keys


def _get_linked_coordinates(
    locdata: pd.DataFrame | pd.Series[Any], coordinate_keys: Iterable[str] | None = None
) -> dict[str, int | float]:
    """
    Combine localization properties from locdata:
    (i) apply weighted averages for spatial coordinates if corresponding
    uncertainties are available;
    (ii) compute uncertainty (as standard deviation with same units as
    coordinates) from weighted average variances for spatial coordinates.

    If coordinate_keys is None, coordinate_keys and corresponding
    coordinate uncertainties from locdata are used.

    If `uncertainty_x` is not available `uncertainty` is taken.
    If no uncertainty is available, unweighted coordinate means are taken.

    Note
    -----
    Uncertainties should have the same unit as coordinates;
    the weights will be 1 / uncertainties^2.

    Parameters
    ----------
    locdata
        dataframe with locdata
    coordinate_keys
        A selection of coordinate keys on which to compute

    Returns
    -------
    dict[str, int | float]
        New position coordinates and related uncertainties as 'uncertainty_c'.
    """
    available_coordinate_keys = _get_loc_property_key_per_dimension(
        locdata=locdata, property_key="position"
    )
    available_uncertainty_keys = _get_loc_property_key_per_dimension(
        locdata=locdata, property_key="uncertainty"
    )
    new_uncertainty_keys = ["uncertainty_x", "uncertainty_y", "uncertainty_z"]

    results_dict = dict()
    for coordinate_key_, uncertainty_key_, new_uncertainty_key_ in zip(
        available_coordinate_keys,
        available_uncertainty_keys,
        new_uncertainty_keys,
    ):
        if coordinate_key_ is not None and (
            coordinate_keys is None or coordinate_key_ in coordinate_keys
        ):
            if not len(locdata):
                return dict()
            elif len(locdata) == 1:
                weighted_mean = locdata[coordinate_key_].iloc[0]
                if uncertainty_key_ is None:
                    weighted_uncertainty = 0
                else:
                    weighted_uncertainty = locdata[uncertainty_key_].iloc[0]
            else:
                if uncertainty_key_ is None:
                    weighted_mean, weighted_variance = weighted_mean_variance(
                        values=locdata[coordinate_key_], weights=None
                    )
                    weighted_uncertainty = np.sqrt(weighted_variance)  # type: ignore
                else:
                    if (locdata[uncertainty_key_] == 0).any():
                        logger.warning(
                            "Zero uncertainties occurred resulting in nan for weighted_mean and weighted_variance."
                        )
                    with np.errstate(invalid="ignore"):
                        weighted_mean, weighted_variance = weighted_mean_variance(
                            values=locdata[coordinate_key_],
                            weights=np.power(
                                1 / locdata[uncertainty_key_], 2
                            ).to_numpy(),  # type: ignore[attr-defined]
                        )
                        weighted_uncertainty = np.sqrt(weighted_variance)  # type: ignore

            results_dict[coordinate_key_] = weighted_mean
            results_dict[new_uncertainty_key_] = weighted_uncertainty

    return results_dict


def _bump_property_key(
    loc_property: str, loc_properties: Iterable[str], extension: str = "_0"
) -> str:
    """
    Add extension to loc_property if loc_property is in loc_properties.
    Repeat recursively until loc_property and all loc_properties are unique.
    """
    if loc_property in loc_properties:
        new_property_label = _bump_property_key(
            loc_property=loc_property + extension,
            loc_properties=loc_properties,
            extension=extension,
        )
    else:
        new_property_label = loc_property
    return new_property_label


def _check_loc_properties(
    locdata: LocData, loc_properties: str | Iterable[str] | None
) -> list[str]:
    """
    Check that loc_properties are valid property keys in locdata.

    Parameters
    ----------
    locdata
        Localization data
    loc_properties
        LocData property keys.
        If None the coordinate_keys of locdata are used.

    Returns
    -------
    list[str]
        Valid localization property keys
    """
    if loc_properties is None:  # use coordinate_keys
        labels = locdata.coordinate_keys.copy()
    elif isinstance(loc_properties, str):
        if loc_properties not in locdata.data.columns:
            raise ValueError(
                f"{loc_properties} is not a valid property in locdata.data."
            )
        labels = [loc_properties]
    else:
        labels = []
        for loc_property in loc_properties:
            if loc_property not in locdata.data.columns:
                raise ValueError(
                    f"{loc_property} is not a valid property in locdata.data."
                )
            labels.append(loc_property)
    return labels


def _dataframe_to_pandas(
    dataframe: DataFrame | None, allow_copy: bool = True
) -> pd.DataFrame | None:
    """
    Convert dataframe that supports the dataframe interchange protocol to
    pandas dataframe.
    """
    if isinstance(dataframe, pd.DataFrame) or dataframe is None:
        return_value = dataframe
    elif not hasattr(pd.api, "interchange"):
        msg = (
            "Use pandas version that implements the DataFrame interchange protocol"
            "or provide pandas.DataFrame object."
        )
        raise TypeError(msg)
    else:
        return_value = pd.api.interchange.from_dataframe(
            dataframe, allow_copy=allow_copy
        )
    return return_value
