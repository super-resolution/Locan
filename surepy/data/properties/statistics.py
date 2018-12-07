"""

Compute statistics for localization data.

These values represent new properties of locdata.

"""
import pandas as pd
from surepy import LocData

def statistics(locdata, statistic_keys = ('count', 'min', 'max', 'mean', 'median', 'std', 'sem')):
    """
    Compute selected statistical parameter for localization data.

    Parameters
    ----------
    locdata : LocData or Pandas DataFrame
        Localization data

    statistic_keys : tuple of strings
        Pandas statistic functions. Default: ('count', 'min', 'max', 'mean', 'median', 'std', 'sem')

    Returns
    -------
    dict
        A dict with descriptive statistics.
    """

    if isinstance(locdata, LocData):
        data = locdata.data
    elif isinstance(locdata, pd.DataFrame):
        data = locdata
    else:
        raise TypeError('locdata should be of type Locdata or Pandas.DataFrame.')

    statistics = data.agg(statistic_keys)
    if isinstance(statistic_keys, str):
        generator = (p for p in list(data))
        dict = {p + '_' + statistic_keys: statistics[p] for p in generator}
    else:
        generator = ((p, s) for p in list(data) for s in statistic_keys)
        dict = {p + '_' + s: statistics[p][s] for p, s in generator}

    return dict
