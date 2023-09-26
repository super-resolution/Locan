"""

Compute statistics for localization data.

These values can represent new properties of locdata.

"""
from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from locan.data.locdata import LocData

__all__: list[str] = ["statistics", "ranges", "range_from_collection"]


def statistics(
    locdata: LocData | pd.DataFrame | pd.Series[Any],
    statistic_keys: str
    | Iterable[str] = ("count", "min", "max", "mean", "median", "std", "sem"),
) -> dict[str, Any]:
    """
    Compute selected statistical parameter for localization data.

    Parameters
    ----------
    locdata
        Localization data

    statistic_keys
        Pandas statistic functions.
        Default: ('count', 'min', 'max', 'mean', 'median', 'std', 'sem')

    Returns
    -------
    dict[str, Any]
        A dict with descriptive statistics.
    """
    data: pd.DataFrame | pd.Series[Any]
    if isinstance(locdata, LocData):
        data = locdata.data
    elif isinstance(locdata, (pd.DataFrame, pd.Series)):
        data = locdata
    else:
        raise TypeError("locdata should be of type locan.LocData or pandas.DataFrame.")

    statistics_ = data.agg(statistic_keys)  # type: ignore

    if isinstance(locdata, pd.Series):
        p = str(data.name)
        if isinstance(statistic_keys, str):
            dict_ = {p + "_" + statistic_keys: statistics_}
        else:
            dict_ = {p + "_" + s: statistics_[s] for s in statistic_keys}  # type: ignore
    else:
        if isinstance(statistic_keys, str):
            generator = (p for p in list(data))
            dict_ = {p + "_" + statistic_keys: statistics_[p] for p in generator}  # type: ignore
        else:
            generator = ((p, s) for p in list(data) for s in statistic_keys)  # type: ignore
            dict_ = {p + "_" + s: statistics_[p][s] for p, s in generator}  # type: ignore

    return dict_


def ranges(
    locdata: LocData,
    loc_properties: str | Iterable[str] | Literal[True] | None = None,
    special: Literal["zero", "link"] | None = None,
    epsilon: float = 1,
) -> npt.NDArray[np.float_] | None:
    """
    Provide data ranges for localization properties.
    If LocData is empty, None is returned.
    If LocData carries a single localization, the range will be
    (value, value + `epsilon`).

    Parameters
    ----------
    locdata
        Localization data.
    loc_properties
        Localization properties for which the range is determined.
        If None the ranges for all spatial coordinates are returned.
        If True the ranges for all localization properties are returned.
    special
        If None (min, max) ranges are determined from data and returned;
        if 'zero' (0, max) ranges with max determined from data are returned.
        if 'link' (min_all, max_all) ranges with min and max determined from
        all combined data are returned.
    epsilon : float
        number to specify the range for single values in locdata.

    Returns
    -------
    npt.NDArray[np.float_] | None
        The data range (min, max) for each localization property.
        Array of shape (dimension, 2).
    """
    if locdata.data.empty:
        return None
    elif len(locdata) == 1:
        pass

    if loc_properties is None:
        ranges_ = locdata.bounding_box.hull.T.copy()  # type: ignore
    elif loc_properties is True:
        ranges_ = np.array([locdata.data.min(), locdata.data.max()]).T
    elif isinstance(loc_properties, str):
        ranges_ = np.array(
            [[locdata.data[loc_properties].min(), locdata.data[loc_properties].max()]]
        )
    else:
        loc_properties = list(loc_properties)
        ranges_ = np.array(
            [locdata.data[loc_properties].min(), locdata.data[loc_properties].max()]
        ).T

    if len(locdata) == 1:
        if ranges_.size == 0:
            ranges_ = np.concatenate(
                [locdata.coordinates, locdata.coordinates + epsilon], axis=0
            ).T
        else:
            ranges_ = ranges_ + [0, epsilon]

    if special is None:
        pass
    elif special == "zero":
        ranges_[:, 0] = 0
    elif special == "link":
        minmax = np.array([ranges_[:, 0].min(axis=0), ranges_[:, 1].max(axis=0)])
        ranges_ = np.repeat(minmax[None, :], len(ranges_), axis=0)
    else:
        raise ValueError(f"The parameter special={special} is not defined.")

    return ranges_


def range_from_collection(
    locdatas: list[LocData],
    loc_properties: str | Iterable[str] | Literal[True] | None = None,
    special: Literal["zero", "link"] | None = None,
    epsilon: float = 1,
) -> tuple[tuple[float, float], ...]:
    """
    Compute the maximum range from all combined localizations for each
    dimension.

    Parameters
    ----------
    locdatas
        Collection of localization datasets.
    loc_properties
        Localization properties for which the range is determined.
        If None the ranges for all spatial coordinates are returned.
        If True the ranges for all locdata.data properties are returned.
    special
        If None (min, max) ranges are determined from data and returned;
        if 'zero' (0, max) ranges with max determined from data are returned.
        if 'link' (min_all, max_all) ranges with min and max determined from
        all combined data are returned.
    epsilon
        number to specify the range for single values in locdata.

    Returns
    -------
    namedtuple
        A namedtuple('Ranges', locdata.coordinate_keys)
        of namedtuple('Range', 'min max').
    """
    ranges_ = [
        ranges(
            locdata=locdata,
            loc_properties=loc_properties,
            special=special,
            epsilon=epsilon,
        )
        for locdata in locdatas
    ]

    mins = np.min([rand[:, 0] for rand in ranges_ if rand is not None], axis=0)
    maxs = np.max([rand[:, 1] for rand in ranges_ if rand is not None], axis=0)

    if loc_properties is None:
        labels = locdatas[0].coordinate_keys
    elif loc_properties is True:
        labels = locdatas[0].data.columns  # type: ignore
    elif isinstance(loc_properties, str):
        labels = [loc_properties]
    else:
        labels = loc_properties  # type: ignore

    Ranges = namedtuple("Ranges", labels)  # type: ignore[misc]
    Range = namedtuple("Range", "min max")
    result = Ranges(
        *(Range(min_value, max_value) for min_value, max_value in zip(mins, maxs))
    )
    return result
