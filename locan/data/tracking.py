"""

Track localizations.

This module provides functions for tracking localizations
(i.e. clustering localization data in time).
The functions take LocData as input and compute new LocData objects.
It makes use of the trackpy package.

"""
from __future__ import annotations

import sys

import pandas as pd

from locan.data.locdata import LocData
from locan.dependencies import HAS_DEPENDENCY, needs_package

if HAS_DEPENDENCY["trackpy"]:
    from trackpy import link_df


__all__ = ["link_locdata", "track"]


@needs_package("trackpy")
def link_locdata(locdata, search_range=40, memory=0, **kwargs) -> pd.Series:
    """
    Track localizations, i.e. cluster localizations in time when nearby in
    successive frames.
    This function applies the trackpy linking method to LocData objects.

    Parameters
    ----------
    locdata : LocData
        Localization data on which to perform the manipulation.
    search_range : float, tuple
        The maximum distance features can move between frames,
        optionally per dimension
    memory : int
        The maximum number of frames during which a feature can vanish,
        then reappear nearby, and be considered the same particle.
    kwargs :
        Other parameters passed to trackpy.link_df().

    Returns
    -------
    pandas.Series
        A series named 'Track' referring to the track number.

    Note
    ----
    In order to switch off the printout from :func:`trackpy.link` and increase
    performance use :func:`trackpy.quiet()` to silence the logging outputs.
    """
    df = link_df(
        locdata.data,
        search_range=search_range,
        memory=memory,
        pos_columns=locdata.coordinate_labels,
        t_column="frame",
        **kwargs,
    )
    return_series = df["particle"]
    return_series.name = "track"
    return return_series


def track(locdata, search_range=40, memory=0, **kwargs) -> tuple[LocData, pd.Series]:
    """
    Cluster (in time) localizations in LocData that are nearby in successive
    frames. Clustered localizations are identified by the trackpy linking
    method.

    The new locdata object carries properties with the same name as the
    original `locata`.
    They are computed as sum for `intensity`, as the first value for `frame`,
    and as mean for all other properties.

    Parameters
    ----------
    locdata : LocData
        Localization data on which to perform the manipulation.
    search_range : float | tuple[float]
        The maximum distance features can move between frames, optionally per
        dimension
    memory : int
        The maximum number of frames during which a feature can vanish, then
        reappear nearby, and be considered the same particle.
    kwargs : dict
        Other parameters passed to trackpy.link_df.

    Returns
    -------
    tuple[Locdata, pandas.Series]
        A new LocData instance assembling all generated selections
        (i.e. localization cluster).
        A series named 'Track' referring to the track number.

    Note
    ----
    In order to switch off the printout from :func:`trackpy.link` and increase
    performance use :func:`trackpy.quiet()` to silence the logging outputs.
    """
    parameter = locals()

    track_series = link_locdata(locdata, search_range, memory, **kwargs)
    grouped = track_series.groupby(track_series)
    selections = [
        LocData.from_selection(locdata=locdata, indices=group.index.values)
        for _, group in grouped
    ]

    for selection in selections:
        for column in locdata.data.columns:
            if "position" not in column and "intensity" != column and "frame" != column:
                column_mean = getattr(selection.data, column).mean()
                selection.properties.update({column: column_mean})
            if column == "intensity":
                intensity_mean = selection.data.intensity.mean()
                selection.properties.update({"intensity_mean": intensity_mean})
                intensity_sum = selection.data.intensity.sum()
                selection.properties.update({"intensity": intensity_sum})
            if column == "frame":
                frame = selection.data.frame.iloc[0]
                selection.properties.update({"frame": frame})

    collection = LocData.from_collection(selections)

    # metadata
    del locdata.meta.history[:]
    locdata.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    return collection, track_series
