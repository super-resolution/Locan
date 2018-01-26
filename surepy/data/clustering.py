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


def clustering_hdbscan(locdata, min_cluster_size = 5, allow_single_cluster = False):
    """
    Cluster localizations in locdata using the hdbscan clustering algorithm.

    Parameters
    ----------
    locdata : LocData
        specifying the localization data on which to perform the manipulation.
    min_cluster_size : int
        minimumm cluster size in HDBSCAN algorithm (default: 5)
    allow_single_cluster : bool
        allowing to return single cluster (default: False)

    Returns
    -------
    LocData
        a new LocData instance assembling all generated selections (i.e. localization cluster).
    """
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        allow_single_cluster=allow_single_cluster,
        gen_min_span_tree=False
    ).fit_predict(locdata.coordinates)

    grouped = locdata.data.groupby(labels)
    selections = list(map(lambda x: LocData.from_selection(locdata=locdata, indices=x), grouped.indices.values()))
    col = LocData.from_collection(*selections)

    # metadata
    col.meta['History'] = list({'Method:': 'clustering_hdbscan', 'Parameter': [min_cluster_size, allow_single_cluster]})

    return col



