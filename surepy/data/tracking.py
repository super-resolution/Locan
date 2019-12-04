"""

Track localizations.

This module provides functions for tracking localizations (i.e. clustering localization data in time).
The functions take LocData as input and compute new LocData objects.
It makes use of the trackpy package.

"""

import sys

import numpy as np
import pandas as pd
try:
    from trackpy import link_df
    _has_trackpy = True
except ImportError:
    _has_trackpy = False

from surepy.data.locdata import LocData


__all__ = ['link_locdata', 'track']


def link_locdata(locdata, search_range=40, memory=0, **kwargs):
    """
    Track localizations, i.e. cluster localizations in time when nearby in successive frames.
    This function applies the trackpy linking method to LocData objects.

    Parameters
    ----------
    locdata : LocData
        Localization data on which to perform the manipulation.
    search_range : float or tuple
        The maximum distance features can move between frames,
        optionally per dimension
    memory : integer, optional
        The maximum number of frames during which a feature can vanish,
        then reappear nearby, and be considered the same particle. 0 by default.

    Other Parameters
    ----------------
    kwargs :
        Other parameters passed to trackpy.link_df().

    Returns
    -------
    pandas Series
        A series named 'Track' referring to the track number.
    """
    if not _has_trackpy:
        raise ImportError("trackpy is required.")

    df = link_df(locdata.data, search_range=search_range, memory=memory, pos_columns=locdata.coordinate_labels,
                    t_column='frame', **kwargs)
    return_series = df['particle']
    return_series.name = 'track'
    return return_series


def track(locdata, search_range=40, memory=0, **kwargs):
    """
    Cluster (in time) localizations in LocData that are nearby in successive frames. Clustered localizations are
    identified by the trackpy linking method.

    Parameters
    ----------
    locdata : LocData
        Localization data on which to perform the manipulation.
    search_range : float or tuple
        The maximum distance features can move between frames, optionally per dimension
    memory : integer, optional
        The maximum number of frames during which a feature can vanish, then reappear nearby,
        and be considered the same particle.

    Other Parameters
    ----------------
    kwargs :
        Other parameters passed to trackpy.link_df.

    Returns
    -------
    Locdata, pandas Series
        A new LocData instance assembling all generated selections (i.e. localization cluster).
        A series named 'Track' referring to the track number.
    """
    parameter = locals()

    track_series = link_locdata(locdata, search_range, memory, **kwargs)
    grouped = track_series.groupby(track_series)
    selections = [LocData.from_selection(locdata=locdata, indices=group.index.values) for _, group in grouped]
    collection = LocData.from_collection(selections)

    # metadata
    del locdata.meta.history[:]
    locdata.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    return collection, track_series