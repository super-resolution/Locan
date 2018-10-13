'''

Methods for tracking localizations (i.e. clustering localization data in time) in LocData objects.

'''

import numpy as np
import pandas as pd
from trackpy import link_df


from sklearn.neighbors import NearestNeighbors
from scipy import stats


from surepy import LocData
from surepy.constants import N_JOBS


def link_trackpy(locdata, search_range=40, memory=0, **kwargs):
    """
    Track localizations, i.e. cluster localizations in time when nearby in successive frames. This function applies the trackpy linking method to LocData objects.

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
    pandas Series
        a series with 'particle' values indicating the track number.
    """
    dat = locdata
    df = link_df(locdata.data, search_range=search_range, memory=memory, pos_columns=locdata.coordinate_labels,
                    t_column='Frame', **kwargs)
    return df['particle']


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
    df = link_trackpy(locdata,search_range, memory, **kwargs)

    grouped = df.groupby(df)

    selections = list(map(lambda x: LocData.from_selection(locdata=locdata, indices=x), grouped.indices.values()))
    collection = LocData.from_collection(*selections)

    # metadata
    del collection.meta.history[:]
    collection.meta.history.add(name='track',
                                parameter='locdata={}, radius={}'.format(
                                    locdata, search_range))

    return collection
