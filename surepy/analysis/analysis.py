"""
This module provides a template for an analysis class.
"""
import time
from surepy.analysis import metadata_analysis_pb2


class Analysis():
    """
    An abstract class for analysis methods to be used on LocData objects.

    The analysis code should go in _compute_results() which is automatically called upon instantiation.

    Parameters
    ----------
    locdata : LocData
        Input data.
    meta : Metadata protobuf message or dictionary
        Metadata about the current analysis routine.
    kwargs : kwarg
        Parameters for the analysis routine.

    Attributes
    -----------
    count : int (class attribute)
        A counter for counting Analysis instantiations.
    locdata : LocData
        reference to the LocData object specified as input data.
    results : pandas data frame or array or array of arrays or None
        The numeric results as derived from the analysis method.
    parameter : dict
        Current parameters for the analysis routine.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    """
    count=0

    def __init__(self, locdata, meta=None, **kwargs):
        """ Provide default atributes."""
        Analysis.count += 1
        self.__class__.count += 1

        self.parameter = kwargs
        self.locdata = locdata
        self.results = self._compute_results(locdata, **kwargs)

        # meta
        self.meta = metadata_analysis_pb2.Metadata()
        self.meta.identifier = str(self.__class__.count)
        self.meta.creation_date = int(time.time())
        self.meta.method.name = str(self.__class__)
        self.meta.method.parameter = str(kwargs)
        #  self.meta.locdata = locdata.meta

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(self.meta, key, value)
        else:
            self.meta.MergeFrom(meta)


    def __del__(self):
        """ updating the counter upon deletion of class instance. """
        Analysis.count -= 1
        self.__class__.count -= 1

    def __str__(self):
        """ Return results in a printable format."""
        return str(self.results)

    def _compute_results(self, locdata, **kwargs):
        """ Apply analysis routine with the specified parameters on locdata and return results."""
        raise NotImplementedError

    def save(self):
        # todo: an appropriate file format needs to be identified.
        """ Save results."""
        raise NotImplementedError

    def save_as_txt(self):
        """ Save results in a text format, that can e.g. serve as Origin import."""
        raise NotImplementedError

    def save_as_yaml(self):
        """ Save results in a text format, that can e.g. serve as Origin import."""
        raise NotImplementedError

    def load(self, results):
        """ Load results."""
        raise NotImplementedError

    def plot(self, ax):
        """ Provide an axes instance with plot of results."""
        # ax.plot(results)
        raise NotImplementedError

    def hist(self, ax):
        """ Provide an axes instance with histogram of results."""
        # ax.hist(results)
        raise NotImplementedError
