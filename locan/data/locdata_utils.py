"""

Utility functions for working with locdata.

"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from locan.constants import PropertyKey
from locan.utils.statistics import weighted_mean_variance

__all__ = []


def _get_available_coordinate_labels(
    locdata: pd.DataFrame | pd.Series, coordinate_labels: Iterable[str] | None = None
) -> tuple[list[str | None], list[str | None]]:
    """
    Get available coordinate_labels paired with uncertainty_labels including
    None if the label for certain dimension is not available.
    If `uncertainty_x` is not available `uncertainty` is taken.

    Parameters
    ----------
    locdata
    coordinate_labels

    Returns
    -------

    """
    available_coordinate_labels = []
    for c_label_ in PropertyKey.coordinate_labels():
        if c_label_ in locdata.columns and (
            coordinate_labels is None or c_label_ in coordinate_labels
        ):
            available_coordinate_labels.append(c_label_)
        else:
            available_coordinate_labels.append(None)

    uncertainty_label = "uncertainty" if "uncertainty" in locdata.columns else None

    uncertainty_labels = []
    for c_label_, u_label_ in zip(
        available_coordinate_labels, ["uncertainty_x", "uncertainty_y", "uncertainty_z"]
    ):
        if c_label_ is None:
            uncertainty_labels.append(None)
        else:
            if u_label_ in locdata.columns:
                uncertainty_labels.append(u_label_)
            else:
                uncertainty_labels.append(uncertainty_label)

    return available_coordinate_labels, uncertainty_labels


def _get_linked_coordinates(locdata, coordinate_labels=None) -> dict[str, int | float]:
    """
    Combine localization properties from locdata:
    (i) apply weighted averages for spatial coordinates if corresponding
    uncertainties are available;
    (ii) compute uncertainty from weighted average variances for spatial
    coordinates

    If coordinate_labels is None, coordinate_labels and corresponding
    coordinate uncertainties from locdata are used.

    If `uncertainty_x` is not availabe `uncertainty` is taken.
    If no uncertainty is available, unweighted coordinate means are taken.

    Parameters
    ----------
    locdata : pd.DataFrame | pd.Series
        dataframe with locdata
    coordinate_labels : Iterable[str] | None
        A selection of coordinate labels on which to compute

    Returns
    -------
    dict[str, int | float]
        New position coordinates and related uncertainties as 'uncertainty_c'.
    """
    available_coordinate_labels, uncertainty_labels = _get_available_coordinate_labels(
        locdata=locdata, coordinate_labels=coordinate_labels
    )

    results_dict = {}
    for coordinate_label_, uncertainty_label_, new_uncertainty_label_ in zip(
        available_coordinate_labels,
        uncertainty_labels,
        ["uncertainty_x", "uncertainty_y", "uncertainty_z"],
    ):
        if coordinate_label_ is not None:
            if not len(locdata):
                return {}
            elif len(locdata) == 1:
                weighted_mean = locdata[coordinate_label_].iloc[0]
                if uncertainty_label_ is None:
                    weighted_variance = 0
                else:
                    weighted_variance = locdata[uncertainty_label_].iloc[0]
            else:
                if uncertainty_label_ is None:
                    weighted_mean, weighted_variance = weighted_mean_variance(
                        values=locdata[coordinate_label_], weights=None
                    )
                else:
                    weighted_mean, weighted_variance = weighted_mean_variance(
                        values=locdata[coordinate_label_],
                        weights=1 / locdata[uncertainty_label_].to_numpy(),
                    )
            results_dict[coordinate_label_] = weighted_mean
            results_dict[new_uncertainty_label_] = weighted_variance

    return results_dict
