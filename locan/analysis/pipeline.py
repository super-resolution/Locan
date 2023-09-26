"""
Building an analysis pipeline.

Pipeline refers to sequential analysis steps that are applied to a single
LocData object.
An analysis pipeline here includes true piped analysis, where a preliminary
result serves as input to the next analysis
step, but also workflows that provide different results in parallel.

A batch process is a procedure for running a pipeline over multiple LocData
objects while collecting and combing
results.

This module provides a class `Pipeline` to combine the analysis procedure,
parameters and results in a single pickleable object.
"""
from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis

if TYPE_CHECKING:
    from locan.data.locdata import LocData

__all__: list[str] = ["Pipeline"]

logger = logging.getLogger(__name__)


class Pipeline(_Analysis):
    """
    The base class for a specialized analysis pipeline to be used on LocData
    objects.

    The custom analysis routine has to be added by implementing the method
    `computation(self, **kwargs)`.
    Keyword arguments must include the locdata reference and optional
    parameters.

    Results are provided as customized attributes.
    We suggest abbreviated standard names for the most common procedures
     such as:

    * lp - Localization Precision
    * lprop - Localization Property
    * lpf - Localizations per Frame
    * rhf - Ripley H function
    * clust - locdata with clustered elements

    Parameters
    ----------
    computation : Callable[..., Any]
        A function `computation(self, **kwargs)` specifying the analysis
        procedure.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    kwargs
        Locdata reference and optional parameters passed to
        `computation(self, **kwargs)`.

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata | None
        Metadata about the current analysis routine.
    computation : Callable[..., Any]
        A function `computation(self, **kwargs)` specifying the analysis
        procedure.
    kwargs
        All parameters including the locdata reference that are passed to
        `computation(self, **kwargs)`.

    Note
    ----
    The class variable `Pipeline.count` is only incremented in a single
    process. In multiprocessing `Pipeline.count` and
    `Pipeline.meta.identifier` (which is set using `count`) cannot be used to
    identify distinct Pipeline objects.

    Note
    ----
    For the Pipeline object to be pickleable attention has to be paid to the
    :func:`computation` method.
    With multiprocessing it will have to be re-injected for each Pipeline
    object by `pipeline.computation = computation`
    after computation and before pickling.
    """

    def __init__(
        self,
        computation: Callable[..., Any],
        meta: metadata_analysis_pb2.AMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)

        if not callable(computation):
            raise TypeError(
                "A callable function `computation(self, locdata, **kwargs)` "
                "must be passed as first argument."
            )
        self.computation = computation
        self.kwargs = kwargs

    def __bool__(self) -> bool:
        return True

    def compute(self) -> Any:
        """
        Run the analysis procedure. All parameters must be given upon Pipeline
        instantiation.
        """
        return self.computation(self, **self.kwargs)

    def save_computation(self, path: str | os.PathLike[Any]) -> None:
        """
        Save the analysis procedure (i.e. the computation() method) as human
        readable text.

        Parameters
        ----------
        path : str | os.PathLike[Any]
            Path and file name for saving the text file.
        """
        with open(path, "w") as handle:
            handle.write(f"Analysis Pipeline: {self.__class__.__name__}\n\n")
            handle.write(inspect.getsource(self.computation))

    def computation_as_string(self) -> str:
        """
        Return the analysis procedure (i.e. the computation() method) as string.
        """
        return inspect.getsource(self.computation)


T_Pipeline = TypeVar("T_Pipeline", bound="Pipeline")


def computation_test(
    self: T_Pipeline,
    locdata: LocData | None = None,
    parameter: str = "test",
) -> T_Pipeline:
    """A pipeline definition for testing."""
    self.locdata = locdata  # type: ignore
    something = "changed_value"
    logger.debug(f"something has a : {something}")
    self.test = parameter  # type: ignore
    logger.info(f"computation finished for locdata: {locdata}")

    try:
        raise NotImplementedError
    except NotImplementedError:
        logger.warning(f"An exception occurred for locdata: {locdata}")

    return self
