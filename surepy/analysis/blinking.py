"""
Compute on- and off-periods from localization frames.

Assuming that the provided localizations are acquired from the same label, we analyze the times of recording as
provided by the `frame`property.


"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from surepy.analysis.analysis_base import _Analysis
from surepy import LocData


import surepy.io.io_locdata as io
from surepy.gui.io import file_dialog
from surepy.data.rois import Roi
from surepy.data.filter import select_by_condition, random_subset, select_by_region
from surepy.analysis import LocalizationPrecision, LocalizationsPerFrame, LocalizationProperty, NearestNeighborDistances, RipleysHFunction
from surepy.render import render_2d
from surepy.data.transform import randomize
from surepy.data.cluster import cluster_hdbscan, cluster_dbscan
from surepy.data.hulls import ConvexHull
from surepy.data.track import track

##### The algorithms


def _blink_statistics(locdata, memory=0, remove_heading_off_periods=True):
    """
    Estimate on and off times from the frame values provided.

    On and off-periods are determined from the sorted frame values.
    A series of frame values that constantly increase by one is considered a on-period.
    Each series of missing frame values between two given frame values is considered an off-period.

    Parameters
    ----------
    locdata : LocData or array-like
        Localization data or just the frame values of given localizations.
    memory : int
        The maximum number of intermittent frames without any localization
        that are still considered to belong to the same on-period.

    Returns
    -------
    tuple of ndarrays
        Two arrays with on- and off-periods in units of frame numbers.
    """
    if isinstance(locdata, LocData):
        frames = locdata.data.frame.values
    else:
        frames = locdata

    frames, counts = np.unique(frames, return_counts=True)

    # provide warning if duplicate frames are found. This should not be the case for appropriate localization clusters.
    if np.any(counts > 1):
        counts_larger_one = counts[counts > 1]
        warnings.warn(f'There are {sum(counts_larger_one) - len(counts_larger_one)} '
                      f'duplicated frames found that will be ignored.')

    # frames are counted from 0. We change this to start with 1 and insert 0 to get a 1 frame on period
    # for a localization in frame 0.
    frames = frames + 1
    frames = np.insert(frames, 0, 0)

    differences = np.diff(frames)
    indices = np.nonzero(differences > memory + 1)[0]
    groups = np.split(differences, indices)

    if groups[0].size == 0:
        groups = groups[1:]

    # on-times
    # the sum is taken to include memory>0.
    # one is added since a single localization is considered to be on for one frame.
    on_periods = np.array([np.sum(group[1:]) + 1 for group in groups])

    # off-times
    off_periods = np.array([group[0] - 1 for group in groups])
    if off_periods[0] == 0:
        off_periods = off_periods[1:]
    else:
        if remove_heading_off_periods:
            off_periods = off_periods[1:]

    return on_periods, off_periods


##### The specific analysis classes

class BlinkStatistics(_Analysis):
    """
    Estimate on and off times from the frame values provided.

    On and off-periods are determined from the sorted frame values.
    A series of frame values that constantly increase by one is considered a on-period.
    Each series of missing frame values between two given frame values is considered an off-period.

    Parameters
    ----------
    memory : int
        The maximum number of intermittent frames without any localization
        that are still considered to belong to the same on-period.
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
    results : tuple of ndarrays
        Two arrays with on- and off-periods in units of frame numbers.
    """
    count = 0

    def __init__(self, meta=None, memory=0, remove_heading_off_periods=True):
        super().__init__(meta, memory=memory, remove_heading_off_periods=remove_heading_off_periods)

    def compute(self, locdata):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData or array-like
            Localization data or just the frame values of given localizations.

        Returns
        -------
        Analysis class
            Returns the Analysis class object (self).
        """
        self.results = _blink_statistics(locdata=locdata, **self.parameter)
        return self
