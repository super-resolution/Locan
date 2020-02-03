"""
Base class that serves as template for a specialized analysis class.

It also provides helper functions to be used in specialized analysis classes.
"""

import time
from surepy.analysis import metadata_analysis_pb2
from scipy import stats


class _Analysis:
    """
    Base class for standard analysis procedures.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.

    Other Parameters
    ----------------
    kwargs :
        Parameter that are passed to the algorithm.

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    locdata : LocData object
        Localization data.
    parameter : dict
        A dictionary with all settings (i.e. the kwargs) for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    """
    count = 0
    """ A counter for counting Analysis class instantiations (class attribute)."""

    def __init__(self, meta, **kwargs):
        self.__class__.count += 1

        self.parameter = kwargs
        self.meta = _init_meta(self)
        self.meta = _update_meta(self, meta)

    def __del__(self):
        """ Update the counter upon deletion of class instance. """
        self.__class__.count -= 1

    def __repr__(self):
        """ Return representation of the Analysis class. """
        return f'{self.__class__.__name__}(**{self.parameter})'

    def compute(self):
        """ Apply analysis routine with the specified parameters on locdata and return results."""
        raise NotImplementedError

    def report(self):
        """ Show a report about analysis results."""
        raise NotImplementedError


# Dealing with metadata

def _init_meta(self):
    meta_ = metadata_analysis_pb2.AMetadata()
    meta_.identifier = str(self.__class__.count)
    meta_.creation_date = int(time.time())
    meta_.method.name = str(self.__class__.__name__)
    meta_.method.parameter = str(self.parameter)
    return meta_


def _update_meta(self, meta=None):
    meta_ = self.meta
    if meta is None:
        pass
    else:
        try:
            meta_.MergeFrom(meta)
        except TypeError:
            for key, value in meta.items():
                setattr(meta_, key, value)

    return meta_


# Dealing with scipy.stats

def _list_parameters(distribution):
    """
    List parameters for scipy.stats.distribution.

    Parameters
    ----------
    distribution : str or scipy.stats distribution object
        Distribution of choice.

    Returns
    -------
    list of str
        A list of distribution parameter strings.
    """
    if isinstance(distribution, str):
        distribution = getattr(stats, distribution)
    if distribution.shapes:
        parameters = [name.strip() for name in distribution.shapes.split(',')]
    else:
        parameters = []
    if distribution.name in stats._discrete_distns._distn_names:
        parameters += ['loc']
    elif distribution.name in stats._continuous_distns._distn_names:
        parameters += ['loc', 'scale']
    else:
        raise TypeError("Distribution name not found in discrete or continuous lists.")

    return parameters
