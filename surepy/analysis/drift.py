"""
Drift analysis for localization coordinates.

This module provides functions for estimating spatial drift in localization data.
"""
import warnings
from itertools import accumulate
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from surepy.constants import _has_open3d
if _has_open3d: import open3d as o3d
from surepy.analysis.analysis_base import _Analysis, _list_parameters
from surepy.data.locdata import LocData
from surepy.data.register import _register_icp_open3d


__all__ = ['Drift']

# todo: class Drift carrying the drift estimate. Show drift as function of frame and fit continuous function.

##### The algorithms

def _estimate_drift_open3d(locdata, chunk_size=1000, target='first'):
    """
    Estimate drift from localization coordinates by registering points in successive time-chunks of localization
    data using an "Iterative Closest Point" algorithm.

    Parameters
    ----------
    locdata : LocData object
       Localization data with properties for coordinates and frame.
    chunk_size : int
       Number of consecutive localizations to form a single chunk of data.
    target : string
       The chunk on which all other chunks are aligned. One of 'first', 'previous'.

    Returns
    -------
    namedtuple of ndarrays
        'matrices' and 'offsets'.
    """
    if not _has_open3d:
        raise ImportError("open3d is required.")

    # split in chunks
    collection = LocData.from_chunks(locdata, chunk_size=chunk_size)

    # register locdatas
    matrices = []
    offsets = []
    if target == 'first':
        for locdata in collection.references[1:]:
            matrix, offset = _register_icp_open3d(locdata.coordinates, collection.references[0].coordinates,
                                                  matrix=None, offset=None, pre_translation=None,
                                                  max_correspondence_distance=100, max_iteration=10_000,
                                                  verbose=False)
            matrices.append(matrix)
            offsets.append(offset)

    elif target == 'previous':
        for n in range(len(collection.references)-1):
            matrix, offset = _register_icp_open3d(collection.references[n+1].coordinates,
                                                  collection.references[n].coordinates,
                                                  matrix=None, offset=None, pre_translation=None,
                                                  max_correspondence_distance=100, max_iteration=10_000,
                                                  with_scaling=False, verbose=False)
            matrices.append(matrix)
            offsets.append(offset)
    results = namedtuple('results', 'collection matrices offsets')
    return results(collection=collection, matrices=np.array(matrices), offsets=np.array(offsets))


##### The specific analysis classes

class Drift(_Analysis):
    """
    Estimate drift.

    Parameters
    ----------
    locdata : LocData object
        Localization data representing the source on which to perform the manipulation.
    chunk_size : int
        Number of consecutive localizations to form a single chunk of data.
    target : string
        The chunk on which all other chunks are aligned. One of 'first', 'previous'.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    collection : Locdata object
        Collection of locdata chunks
    results : namedtuple of ndarrays
        'matrices' and 'offsets'.
    """
    count = 0

    def __init__(self, meta=None, chunk_size=1000, target='first'):
        super().__init__(meta, chunk_size=chunk_size, target=target)
        self.collection = None
        self.results = None

    def compute(self, locdata):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData object
            Localization data representing the source on which to perform the manipulation.

        Returns
        -------
        Analysis class
            Returns the Analysis class object (self).
        """

        drift_results = _estimate_drift_open3d(locdata=locdata, **self.parameter)
        self.collection = drift_results.collection
        results = namedtuple('results', 'matrices offsets')
        self.results = results(matrices=drift_results.matrices, offsets=drift_results.offsets)
        return self

    def plot(self, ax=None, results_field='matrices', element=(0, 0), window=1, **kwargs):
        """
        Provide plot as matplotlib axes object showing the running average of results over window size.

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image
        element : string
            The element of results to be plotted; if None all plots are shown.
            One of 'matrices[:, 0, 0]' or 'offsets[:, 0]
        window: int
            Window for running average that is applied before plotting.
            Not implemented yet.

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to matplotlib.pyplot.plot().

        Returns
        -------
        matplotlib Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        n_transformations = len(self.collection)-1
        # prepare plot
        x = [reference.data.frame.mean() for reference in self.collection.references[1:]]
        if element is None:
            ys = getattr(self.results, results_field).reshape(n_transformations, -1).T
            for y in ys:
                ax.plot(x, y, **kwargs)
        else:
            y = getattr(self.results, results_field).reshape(n_transformations, -1).T[element]
            ax.plot(x, y, **kwargs)

        ax.set(title=f'Drift\n (window={window})',
               xlabel='frame',
               ylabel=''.join([results_field, '[', str(element), ']'])
               )

        return ax
