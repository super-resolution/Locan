"""
Base class that serves as template for a specialized analysis class.

It also provides helper functions to be used in specialized analysis classes.
"""

import time
from surepy.analysis import metadata_analysis_pb2


class _Analysis():
    """
    Base class for standard analysis procedures.

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
        A dictionary with all settings (i.e. the kwargs) for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    """
    count = 0

    def __init__(self, locdata, meta, **kwargs):
        self.__class__.count += 1

        self.locdata = locdata
        self.parameter = kwargs
        self.meta = _init_meta(self)
        self.meta = _update_meta(self, meta)

    def __del__(self):
        """ Update the counter upon deletion of class instance. """
        self.__class__.count -= 1

    def compute(self):
        """ Apply analysis routine with the specified parameters on locdata and return results."""
        raise NotImplementedError

    def save(self, path):
        """ Save Analysis object."""
        raise NotImplementedError

    def load(self, path):
        """ Load Analysis object."""
        raise NotImplementedError

    def report(self, ax):
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

