"""
Base class that serves as template for a specialized analysis class.

It also provides helper functions to be used in specialized analysis classes.
"""
import time

from google.protobuf import text_format, json_format
from scipy import stats

from locan.analysis import metadata_analysis_pb2
from locan.utils.format import _time_string


class _Analysis:
    """
    Base class for standard analysis procedures.

    Parameters
    ----------
    locdata : LocData
        Localization data.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    kwargs :
        Parameter that are passed to the algorithm.

    Attributes
    ----------
    locdata : LocData
        Localization data.
    parameter : dict
        A dictionary with all settings (i.e. the kwargs) for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : any
        Computed results.
    """
    count = 0
    """int: A counter for counting Analysis class instantiations (class attribute)."""

    def __init__(self, meta, **kwargs):
        self.__class__.count += 1

        self.parameter = kwargs
        self.meta = _init_meta(self)
        self.meta = _update_meta(self, meta)
        self.results = None

    def __del__(self):
        """ Update the counter upon deletion of class instance. """
        self.__class__.count -= 1

    def __repr__(self):
        """ Return representation of the Analysis class. """
        parameter_string = ", ".join((f'{key}={value}' for key, value in self.parameter.items()))
        return f'{self.__class__.__name__}({parameter_string})'

    def __getstate__(self):
        """Modify pickling behavior."""
        # Copy the object's state from self.__dict__ to avoid modifying the original state.
        state = self.__dict__.copy()
        # Serialize the unpicklable protobuf entries.
        json_string = json_format.MessageToJson(self.meta, including_default_value_fields=False)
        state['meta'] = json_string
        return state

    def __setstate__(self, state):
        """Modify pickling behavior."""
        # Restore instance attributes.
        self.__dict__.update(state)
        # Restore protobuf class for meta attribute
        self.meta = metadata_analysis_pb2.AMetadata()
        self.meta = json_format.Parse(state['meta'], self.meta)

    def __bool__(self):
        if self.results is not None:
            return True
        else:
            return False

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
    meta_.creation_date = _time_string(time.time())
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
    distribution : str, scipy.stats.distribution
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
