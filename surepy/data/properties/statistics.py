

def statistics(locdata, statistic_keys = ('count', 'min', 'max', 'mean', 'median', 'std', 'sem')):
    """
    Compute statistics for localization data.

    Parameters
    ----------
    locdata : LocData
        localization data

    statistic_keys : tuple of strings
        pandas statistic functions. Default: ('count', 'min', 'max', 'mean', 'median', 'std', 'sem')

    Returns
    -------
    dict
        A dict with descriptive statistics.
    """

    statistics = locdata.data.agg(statistic_keys)
    generator = ((p, s) for p in list(locdata.data) for s in statistic_keys)
    dict = {p + '_' + s: statistics[p][s] for p, s in generator}

    return dict
