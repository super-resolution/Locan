"""

Compute statistics for localization data.

These values represent new properties of locdata.

"""
import pandas as pd
from surepy.data.locdata import LocData


__all__ = ['statistics']


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

    statistics = data.agg(statistic_keys)

    if isinstance(locdata, pd.Series):
        p = data.name
        if isinstance(statistic_keys, str):
            dict = {p + '_' + statistic_keys: statistics}
        else:
            dict = {p + '_' + s: statistics[s] for s in statistic_keys}
    else:
        if isinstance(statistic_keys, str):
            generator = (p for p in list(data))
            dict = {p + '_' + statistic_keys: statistics[p] for p in generator}
        else:
            generator = ((p, s) for p in list(data) for s in statistic_keys)
            dict = {p + '_' + s: statistics[p][s] for p, s in generator}

    return dict
