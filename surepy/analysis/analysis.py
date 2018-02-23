"""
This module provides a template for a specialized analysis class.
It also provides helper functions to be used in specialized analysis classes.
And it provides standard interface functions to be used in specialized analysis classes.
"""
import time
from surepy.analysis import metadata_analysis_pb2


class Analysis():
    """
    The base class for specialized analysis classes to be used on LocData objects.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    kwargs :
        Parameter that are passed to the algorithm.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    locdata : LocData object
        Localization data.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : numpy array or pandas DataFrame
        Computed results.
    """
    count = 0

    def __init__(self, locdata, meta, **kwargs):
        Analysis.count += 1
        self.__class__.count += 1

        self.locdata = locdata
        self.parameter = kwargs
        self.meta = self._init_meta(meta=meta)
        self.results = None

    def _init_meta(self, meta=None):
        meta_ = metadata_analysis_pb2.AMetadata()
        meta_.identifier = str(self.__class__.count)
        meta_.creation_date = int(time.time())
        meta_.method.name = str(self.__class__.__name__)
        meta_.method.parameter = str(self.parameter)

        if meta is None:
            pass
        else:
            try:
                meta_.MergeFrom(meta)
            except TypeError:
                for key, value in meta.items():
                    setattr(meta_, key, value)

        return meta_


    def __del__(self):
        """ updating the counter upon deletion of class instance. """
        Analysis.count -= 1
        self.__class__.count -= 1

    def __str__(self):
        """ Return results in a printable format."""
        return str(self.results)

    def compute(self):
        """ Apply analysis routine with the specified parameters on locdata and return results."""
        raise NotImplementedError

    def save(self, path):
        """ Save Analysis object."""
        raise NotImplementedError

    def load(self, path):
        """ Load Analysis object."""
        raise NotImplementedError

    def save_results(self, path):
        """ Save results in a text format, that can e.g. serve as Origin import."""
        raise NotImplementedError

    def plot(self, ax):
        """ Provide an axes instance with plot of results."""
        raise NotImplementedError

    def report(self, ax):
        """ Show a report about analysis results."""
        raise NotImplementedError


