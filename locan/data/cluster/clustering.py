"""

Methods for clustering localization data in LocData objects.

"""

import sys
from copy import copy, deepcopy

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from locan import HAS_DEPENDENCY, needs_package

if HAS_DEPENDENCY["hdbscan"]:
    from hdbscan import HDBSCAN

from locan.configuration import N_JOBS
from locan.data.locdata import LocData

__all__ = ["cluster_hdbscan", "cluster_dbscan"]


@needs_package("hdbscan")
def cluster_hdbscan(
    locdata,
    min_cluster_size=5,
    loc_properties=None,
    allow_single_cluster=False,
    **kwargs
):
    """
    Cluster localizations in locdata using the hdbscan clustering algorithm.

    Parameters
    ----------
    locdata : LocData
        specifying the localization data on which to perform the manipulation.
    loc_properties : list of string, None
        The LocData properties to be used for clustering. If None, locdata.coordinates will be used.
    min_cluster_size : int
        minimumm cluster size in HDBSCAN algorithm (default: 5)
    allow_single_cluster : bool
        allowing to return single cluster (default: False)
    kwargs : dict
        Other parameters passed to `hdbscan.HDBSCAN`.

    Returns
    -------
    tuple (LocData, LocData)
        A tuple with noise and cluster.
        The first LocData object is a selection of all localizations that are defined as noise,
        in other words all localizations that are not part of any cluster.
        The second LocData object is a LocData instance assembling all generated selections (i.e. localization cluster).
    """
    parameter = locals()

    if len(locdata) == 0:
        locdata_noise = LocData()
        collection = LocData()

    if len(locdata) < min_cluster_size:
        locdata_noise = copy(locdata)
        collection = LocData()

    else:
        if loc_properties is None:
            fit_data = locdata.coordinates
        else:
            fit_data = locdata.data[loc_properties]

        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            allow_single_cluster=allow_single_cluster,
            gen_min_span_tree=False,
            **kwargs
        ).fit_predict(fit_data)

        grouped = locdata.data.groupby(labels)
        locdata_index_labels = [
            locdata.data.index[idxs] for idxs in grouped.indices.values()
        ]
        selections = [
            LocData.from_selection(locdata=locdata, indices=idxs)
            for idxs in locdata_index_labels
        ]

        try:
            grouped.get_group(-1)
            locdata_noise = selections[0]
            collection = LocData.from_collection(selections[1:])
        except KeyError:
            locdata_noise = None
            collection = LocData.from_collection(selections)

    # set regions
    if locdata_noise:
        locdata_noise.region = locdata.region
    if collection:
        collection.region = locdata.region

    # metadata
    if locdata_noise:
        del locdata_noise.meta.history[:]
        locdata_noise.meta.history.add(
            name=sys._getframe().f_code.co_name, parameter=str(parameter)
        )
    del collection.meta.history[:]
    collection.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    return locdata_noise, collection


def cluster_dbscan(locdata, eps=20, min_samples=5, loc_properties=None, **kwargs):
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
    loc_properties : list of string, None
        The LocData properties to be used for clustering. If None, locdata.coordinates will be used.
    kwargs : dict
        Other parameters passed to `sklearn.cluster.DBSCAN`.

    Returns
    -------
    tuple (LocData, LocData)
        A tuple with noise and cluster.
        The first LocData object is a selection of all localizations that are defined as noise,
        in other words all localizations that are not part of any cluster.
        The second LocData object is a LocData instance assembling all generated selections (i.e. localization cluster).
    """
    parameter = locals()

    if len(locdata) == 0:
        locdata_noise = LocData()
        collection = LocData()

    else:
        if loc_properties is None:
            fit_data = locdata.coordinates
        else:
            fit_data = locdata.data[loc_properties]

        labels = DBSCAN(
            eps=eps, min_samples=min_samples, n_jobs=N_JOBS, **kwargs
        ).fit_predict(fit_data)

        grouped = locdata.data.groupby(labels)
        locdata_index_labels = [
            locdata.data.index[idxs] for idxs in grouped.indices.values()
        ]
        selections = [
            LocData.from_selection(locdata=locdata, indices=idxs)
            for idxs in locdata_index_labels
        ]

        try:
            grouped.get_group(-1)
            locdata_noise = selections[0]
            collection = LocData.from_collection(selections[1:])
        except KeyError:
            locdata_noise = None
            collection = LocData.from_collection(selections)

    # set regions
    if locdata_noise:
        locdata_noise.region = locdata.region
    if collection:
        collection.region = locdata.region

    # metadata
    if locdata_noise:
        del locdata_noise.meta.history[:]
        locdata_noise.meta.history.add(
            name=sys._getframe().f_code.co_name, parameter=str(parameter)
        )
    del collection.meta.history[:]
    collection.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    return locdata_noise, collection
