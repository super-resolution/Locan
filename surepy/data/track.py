'''

Methods for tracking localizations (i.e. clustering localization data in time) in LocData objects.

'''

import sys

import numpy as np
import pandas as pd
from trackpy import link_df

from surepy import LocData


def link_locdata(locdata, search_range=40, memory=0, **kwargs):
    """
    Track localizations, i.e. cluster localizations in time when nearby in successive frames.
    This function applies the trackpy linking method to LocData objects.

    Parameters
    ----------
    locdata : LocData
        specifying the localization data on which to perform the manipulation.
    search_range : float or tuple
        the maximum distance features can move between frames,
        optionally per dimension
    memory : integer, optional
        the maximum number of frames during which a feature can vanish,
        then reappear nearby, and be considered the same particle. 0 by default.
   kwargs :
       Parameters passed to trackpy.link_df().

    Returns
    -------
    pandas DataFrame
        A DataFrame with 'Index' referring to the locdata indices and 'Track' values indicating the track number.
    """
    df = link_df(locdata.data, search_range=search_range, memory=memory, pos_columns=locdata.coordinate_labels,
                    t_column='Frame', **kwargs)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Index', 'particle': 'Track'}, inplace=True)
    return df[['Index', 'Track']]


def track(locdata, search_range=40, memory=1, **kwargs):
    """
    Cluster (in time) localizations in LocData that are nearby in successive frames. Clustered localizations are
    identified by the trackpy linking method.

    Parameters
    ----------
    locdata : LocData
        specifying the localization data on which to perform the manipulation.
    search_range : float or tuple
        the maximum distance features can move between frames, optionally per dimension
    memory : integer, optional
        the maximum number of frames during which a feature can vanish, then reappear nearby,
        and be considered the same particle. 0 by default.
   kwargs :
       Parameters passed to trackpy.link_df.

    Returns
    -------
    Locdata
        a new LocData instance assembling all generated selections (i.e. localization cluster).
    """
    parameter = locals()

    df = link_locdata(locdata,search_range, memory, **kwargs)

    grouped = df.groupby('Track')

    selections = [LocData.from_selection(locdata=locdata, indices=group['Index'].values) for _, group in grouped]
    collection = LocData.from_collection(*selections)

    # metadata
    del locdata.meta.history[:]
    locdata.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    return collection
