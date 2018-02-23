"""
This module provides a template for a specialized analysis class.
It also provides helper functions to be used in specialized analysis classes.
And it provides standard interface functions to be used in specialized analysis classes.
"""
import time
from surepy.analysis import metadata_analysis_pb2


class Analysis():
    """
    An (abstract) class for analysis methods to be used on LocData objects.

    This class only serves for illustration of a typical specialized analysis class and provides names for typical
    interface functions.

    The interface functions are implemented in stand-alone functions that are called from a particular Analysis method.
    """
    count=0

    def __init__(self):
        Analysis.count += 1
        self.__class__.count += 1

        self.locdata = None
        self.algorithm = None
        self.params = None
        self.meta = None
        self.results = None

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


#### helper functions

    def _init_meta(self, meta=None):
        '''
        Initializes metadata for analysis method from standard settings and user input.

        Parameter
        ---------
        meta : Metadata protobuf message or dict
            Metadata about the current analysis routine.

        Returns:
        Metadata protobuf message
            Metadata about the current analysis routine.
        '''
        meta_ = metadata_analysis_pb2.AMetadata()
        meta_.identifier = str(self.__class__.count)
        meta_.creation_date = int(time.time())
        meta_.method.name = str(self.__class__.__name__)
        meta_.method.algorithm = str(self.algorithm.__name__)
        meta_.method.parameter = str(self.params)

        if meta is None:
            pass
        else:
            try:
                meta_.MergeFrom(meta)
            except TypeError:
                for key, value in meta.items():
                    setattr(meta_, key, value)

        return meta_

