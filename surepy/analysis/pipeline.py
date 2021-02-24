"""
Building an analysis pipeline.

Pipeline refers to sequential analysis steps that are applied to a single LocData object.
An analysis pipeline here includes true piped analysis, where a preliminary result serves as input to the next analysis
step, but also workflows that provide different results in parallel.

A batch process is a procedure for running a pipeline over multiple LocData objects while collecting and combing
results.

This module provides a class `Pipeline` to combine the analysis procedure, parameters and results
in a single pickleable object.
"""
import inspect
import logging

from google.protobuf import text_format, json_format

from surepy.data.locdata import LocData
from surepy.io.io_locdata import load_locdata
from surepy.data.rois import Roi
from surepy.data.hulls import ConvexHull
from surepy.data.filter import select_by_condition
from surepy.data.cluster.clustering import cluster_hdbscan
from surepy.analysis.analysis_base import _init_meta, _update_meta
from surepy.analysis import metadata_analysis_pb2


__all__ = ['Pipeline']

logger = logging.getLogger(__name__)


class Pipeline:
    """
    The base class for a specialized analysis pipeline to be used on LocData objects.

    The custom analysis routine has to be added by implementing the method `computation(self, **kwargs)`.
    Keyword arguments must include the locdata reference and optional parameters.

    Results are provided as customized attributes.
    We suggest abbreviated standard names for the most common procedures such as:

    * lp - Localization Precision
    * lprop - Localization Property
    * lpf - Localizations per Frame
    * rhf - Ripley H function
    * clust - locdata with clustered elements

    Parameters
    ----------
    computation : callable
        A function `computation(self, **kwargs)` specifying the analysis procedure.
    meta : surepy.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    kwargs : dict
        Locdata reference and optional parameters passed to `computation(self, **kwargs)`.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    computation : callable
        A function `computation(self, **kwargs)` specifying the analysis procedure.
    meta : surepy.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    parameter : dict
        All parameters including the locdata reference that were passed to `computation(self, **kwargs)`.

    Notes
    -----
    The class variable `Pipeline.count` is only incremented in a single process. In multiprocessing `Pipeline.count` and
    `Pipeline.meta.identifier` (which is set using `count`) cannot be used to identify distinct Pipeline objects.

    Notes
    -----
    For the Pipeline object to be pickleable attention has to be paid to the :func:`computation` method.
    With multiprocessing it will have to be re-injected for each Pipeline object by `pipeline.computation = computation`
    after computation and before pickling.
    """
    # todo sc_check on input of meta

    count = 0

    def __init__(self, computation, meta=None, **kwargs):
        self.__class__.count += 1

        if not callable(computation):
            raise TypeError('A callable function `computation(self, locdata, **kwargs)` '
                            'must be passed as first argument.')
        self.computation = computation
        self.parameter = kwargs  # is needed to init metadata_analysis_pb2.
        self.meta = _init_meta(self)
        self.meta = _update_meta(self, meta)

    def __del__(self):
        """ updating the counter upon deletion of class instance. """
        self.__class__.count -= 1

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

    def compute(self):
        """ Run the analysis procedure. All parameters must be given upon Pipeline instantiation."""
        return self.computation(self, **self.parameter)

    def save_computation(self, path):
        """
        Save the analysis procedure (i.e. the computation() method) as human readable text.

        Parameters
        ----------
        path : str, os.PathLike
            Path and file name for saving the text file.
        """
        with open(path, 'w') as handle:
            handle.write('Analysis Pipeline: {}\n\n'.format(self.__class__.__name__))
            handle.write(inspect.getsource(self.computation))

    def computation_as_string(self):
        """
        Return the analysis procedure (i.e. the computation() method) as string.
        """
        return inspect.getsource(self.computation)


def computation_test(self, locdata=None, parameter='test'):
    """ A pipeline definition for testing."""
    self.locdata = locdata
    something = 'changed_value'
    logger.debug(f'something has a : {something}')
    self.test = parameter
    logger.info(f'computation finished for locdata: {locdata}')

    try:
        raise NotImplementedError
    except NotImplementedError:
        logger.warning(f'An exception occured for locdata: {locdata}')

    return self
