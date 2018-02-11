'''

Methods for clustering localization data in LocData objects..

'''

import numpy as np
import pandas as pd
import hdbscan
from surepy import LocData


def clustering(locdata, **kwargs):
    """
    Cluster localizations in LocData using a specified clustering algorithm.

    Parameters
    ----------
    locdata : LocData
        specifying the localization data on which to perform the manipulation.
    kwargs : dict
        Parameters for specified clustering algorithm.

    Returns
    -------
    Collection
        a new instance of Collection assembling all generated selections (i.e. localization cluster).
    """
    raise NotImplementedError


def clustering_hdbscan(locdata, min_cluster_size = 5, kdims=None, allow_single_cluster = False, kdims=False noise=False):
    """
    Cluster localizations in locdata using the hdbscan clustering algorithm.

    Parameters
    ----------
    locdata : LocData
        specifying the localization data on which to perform the manipulation.
    kdims : list of Propery keywords to be used for clustering. If None, locdata.coordinates will be used
    min_cluster_size : int
        minimumm cluster size in HDBSCAN algorithm (default: 5)
    allow_single_cluster : bool
        allowing to return single cluster (default: False)
    noise : bool
        Flag indicating if the first cluster represents noise. If True a tuple of LocData objects is returned with
        noise and cluster collection. If False a single LocData object is returned.

    Returns
    -------
    LocData or tuple of LocData
        A new LocData instance assembling all generated selections (i.e. localization cluster).
        If noise is True the first LocData object is a selection of all localizations that are defined as noise.
    """
    if kdims is None:
        fit_data = locdata.coordinates
    else:
        fit_data = locdata.data[kdims]
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        allow_single_cluster=allow_single_cluster,
        gen_min_span_tree=False
    ).fit_predict(fitdata)

    grouped = locdata.data.groupby(labels)

    if noise:
        selections = list(map(lambda x: LocData.from_selection(locdata=locdata, indices=x), grouped.indices.values()))
        noise = selections[0]
        collection = LocData.from_collection(*selections[1:])
    else:
        selections = list(map(lambda x: LocData.from_selection(locdata=locdata, indices=x), grouped.indices.values()))
        collection = LocData.from_collection(*selections)

    # metadata
    del collection.meta.history[:]
    collection.meta.history.add(name='clustering_hdbscan',
                         parameter='locdata={}, min_cluster_size={}, allow_single_cluster={}'.format(
                             locdata, min_cluster_size, allow_single_cluster))

    if noise:
        return noise, collection
    else:
        return collection



