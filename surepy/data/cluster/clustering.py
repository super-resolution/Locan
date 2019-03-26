'''

Methods for clustering localization data in LocData objects.

'''

import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats

from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN

from surepy import LocData
from surepy.constants import N_JOBS

# A general function for clustering should maybe be implemented.
#
# def find_cluster(locdata, algorithm, **kwargs):
#     """
#     Cluster localizations in LocData using a specified clustering algorithm.
#
#     Parameters
#     ----------
#     locdata : LocData
#         Localization data on which to perform the manipulation.
#     algorithm : str
#         Name of the cluster function to use on `locdata`.
#
#     Other Parameters
#     ----------------
#     kwargs : dict
#         Parameters for specified clustering algorithm.
#
#     Returns
#     -------
#     LocData
#         A new instance of LocData (representing a collection of Locdata)
#         assembling all generated selections (i.e. localization cluster).
#     """
#     raise NotImplementedError


def cluster_hdbscan(locdata, min_cluster_size = 5, kdims=None, allow_single_cluster = False, noise=False, **kwargs):
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

    Other Parameters
    ----------------
    kwargs : dict
        Other parameters passed to `hdbscan.HDBSCAN`.

    Returns
    -------
    LocData or tuple of LocData
        A new LocData instance assembling all generated selections (i.e. localization cluster).
        If noise is True the first LocData object is a selection of all localizations that are defined as noise.
    """
    parameter = locals()

    if kdims is None:
        fit_data = locdata.coordinates
    else:
        fit_data = locdata.data[kdims]

    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        allow_single_cluster=allow_single_cluster,
        gen_min_span_tree=False,
        **kwargs
    ).fit_predict(fit_data)

    grouped = locdata.data.groupby(labels)

    if noise:
        selections = list(map(lambda x: LocData.from_selection(locdata=locdata, indices=x), grouped.indices.values()))
        noise = selections[0]
        collection = LocData.from_collection(selections[1:])
    else:
        selections = list(map(lambda x: LocData.from_selection(locdata=locdata, indices=x), grouped.indices.values()))
        collection = LocData.from_collection(selections)

    # set regions
    if noise:
        noise.region = locdata.region
    collection.region = locdata.region

    # metadata
    if noise:
        del noise.meta.history[:]
        noise.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))
    del collection.meta.history[:]
    collection.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    if noise:
        return noise, collection
    else:
        return collection


def cluster_dbscan(locdata, eps=20, min_samples=5, kdims=None, noise=False, **kwargs):
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

    Other Parameters
    ----------------
    kwargs : dict
        Other parameters passed to `sklearn.cluster.DBSCAN`.

    Returns
    -------
    LocData or tuple of LocData
        A new LocData instance assembling all generated selections (i.e. localization cluster).
        If noise is True the first LocData object is a selection of all localizations that are defined as noise.
    """
    parameter = locals()

    if kdims is None:
        fit_data = locdata.coordinates
    else:
        fit_data = locdata.data[kdims]

    labels = DBSCAN(
        eps=eps, min_samples=min_samples, n_jobs=N_JOBS, **kwargs
    ).fit_predict(fit_data)

    grouped = locdata.data.groupby(labels)

    if noise:
        selections = list(map(lambda x: LocData.from_selection(locdata=locdata, indices=x), grouped.indices.values()))
        noise = selections[0]
        collection = LocData.from_collection(selections[1:])
    else:
        selections = list(map(lambda x: LocData.from_selection(locdata=locdata, indices=x), grouped.indices.values()))
        collection = LocData.from_collection(selections)

    # set regions
    if noise:
        noise.region = locdata.region
    collection.region = locdata.region

    # metadata
    if noise:
        del noise.meta.history[:]
        noise.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))
    del collection.meta.history[:]
    collection.meta.history.add(name=sys._getframe().f_code.co_name, parameter=str(parameter))

    if noise:
        return noise, collection
    else:
        return collection
