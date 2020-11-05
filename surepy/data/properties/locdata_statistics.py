"""

Compute statistics for localization data.

These values can represent new properties of locdata.

"""
from collections import namedtuple

import numpy as np
import pandas as pd

from surepy.data.locdata import LocData


__all__ = ['statistics', 'range_from_collection']


def statistics(locdata, statistic_keys=('count', 'min', 'max', 'mean', 'median', 'std', 'sem')):
    """
    Compute selected statistical parameter for localization data.

    Parameters
    ----------
    locdata : LocData, Pandas DataFrame or Pandas Series
        Localization data

    statistic_keys : str or tuple of strings
        Pandas statistic functions. Default: ('count', 'min', 'max', 'mean', 'median', 'std', 'sem')

    Returns
    -------
    dict
        A dict with descriptive statistics.
    """

    if isinstance(locdata, LocData):
        data = locdata.data
    elif isinstance(locdata, (pd.DataFrame, pd.Series)):
        data = locdata
    else:
        raise TypeError('locdata should be of type Locdata or Pandas.DataFrame.')

    statistics_ = data.agg(statistic_keys)

    if isinstance(locdata, pd.Series):
        p = data.name
        if isinstance(statistic_keys, str):
            dict_ = {p + '_' + statistic_keys: statistics_}
        else:
            dict_ = {p + '_' + s: statistics_[s] for s in statistic_keys}
    else:
        if isinstance(statistic_keys, str):
            generator = (p for p in list(data))
            dict_ = {p + '_' + statistic_keys: statistics_[p] for p in generator}
        else:
            generator = ((p, s) for p in list(data) for s in statistic_keys)
            dict_ = {p + '_' + s: statistics_[p][s] for p, s in generator}

    return dict_


def range_from_collection(locdata):
    """
    Compute the maximum range from all combined localizations for each dimension.

    Parameters
    ----------
    locdata : LocData or list(LocData)
        Collection of localization datasets.

    Returns
    -------
    namedtuple
        A namedtuple('Ranges', locdata.coordinate_labels) of namedtuple('Range', 'min max').
    """
    if isinstance(locdata.references, list):
        locdatas = locdata.references
    elif isinstance(locdata, (list, tuple)):
        locdatas = locdata
    else:
        raise TypeError('locdata must contain a collection of Locdata objects.')

    ranges = [locdata.bounding_box.hull for locdata in locdatas]
    mins = np.array(ranges)[:, 0].min(axis=0)
    maxs = np.array(ranges)[:, 1].max(axis=0)

    Ranges = namedtuple('Ranges', locdata.coordinate_labels)
    Range = namedtuple('Range', 'min max')
    result = Ranges(*(Range(min_value, max_value) for min_value, max_value in zip(mins, maxs)))
    return result
