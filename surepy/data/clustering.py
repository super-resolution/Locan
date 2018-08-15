'''

Methods for clustering localization data in LocData objects..

'''

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN

from surepy import LocData
from surepy.constants import N_JOBS


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


def clustering_hdbscan(locdata, min_cluster_size = 5, kdims=None, allow_single_cluster = False, noise=False):
    """
    Cluster localizations in locdata using the hdbscan clustering algorithm.

    Parameters
    ----------
    locdata : LocData
        specifying the localization data on which to perform the manipulation.
    kdims : list of string
        Propery keywords to be used for clustering. If None, locdata.coordinates will be used.
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

    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        allow_single_cluster=allow_single_cluster,
        gen_min_span_tree=False
    ).fit_predict(fit_data)

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
                         parameter='locdata={}, min_cluster_size={}, kdims={}, allow_single_cluster={}'.format(
                             locdata, min_cluster_size, kdims, allow_single_cluster))

    if noise:
        return noise, collection
    else:
        return collection



def clustering_dbscan(locdata, eps=20, min_samples=5, kdims=None, noise=False):
    """
    Cluster localizations in locdata using the dbscan clustering algorithm as implemented in sklearn.

    Parameters
    ----------
    locdata : LocData
        specifying the localization data on which to perform the manipulation.
    eps : float
        The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as a core point.
        This includes the point itself.
    kdims : list of string
        Propery keywords to be used for clustering. If None, locdata.coordinates will be used.
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

    labels = DBSCAN(
        eps=eps, min_samples=min_samples, metric='euclidean', metric_params=None, algorithm='auto',
        leaf_size=30, p=None, n_jobs=N_JOBS
    ).fit_predict(fit_data)

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
    collection.meta.history.add(name='clustering_dbscan',
                         parameter='locdata={}, eps={}, min_samples={}, kdims={}'.format(
                             locdata, eps, min_samples, kdims))

    if noise:
        return noise, collection
    else:
        return collection


