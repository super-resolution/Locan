"""
Example for a specialized analysis class.

It includes two algorithms for specific analysis routines.
And it provides standard interface functions modified for the specific analysis routine like report.
"""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis
from locan.locan_types import RandomGeneratorSeed

if TYPE_CHECKING:
    import matplotlib as mpl

    from locan.data.locdata import LocData

#
# The algorithms
#
# First there is an algorithm to compute something given a set of point coordinates.
# Second there might be an alternative algorithm doing the same thing.
# Both can be tested with some simple point data.
#
# The user will not have to use this particular function directly.
#


def _algorithm_1(
    data: Any | None = None, limits: tuple[int, int] = (0, 10)
) -> pd.DataFrame:
    """Provides a list of data values. data would be input data that is currently not used."""
    results = [i for i in range(*limits)]  # some complicated algorithm
    results_df = pd.DataFrame.from_dict(dict(a=results))
    return results_df


def _algorithm_2(
    data: Any | None = None, n_sample: int = 100, seed: RandomGeneratorSeed = None
) -> pd.DataFrame:
    """Provides random normal distributed data. data would be input data that is currently not used."""
    rng = np.random.default_rng(seed)
    dict_ = {"a": rng.normal(size=n_sample), "b": rng.normal(size=n_sample)}
    results = pd.DataFrame.from_dict(dict_)
    return results


#
# The base analysis class
#
# Now we want a class implementing this algorithm to be used with locdata. Also the results from this algorithm should
# be reused in some visual representation that is specific for this analysis routine.
# Therefore we have a class that holds results and organizes metadata and provides the specific plotting routine.
#

# class: analysis.analysis_base._Analysis


#
# This specific analysis classes inherit from _Analysis.
#
# The classes for each particular algorithm are defined as:
#


class AnalysisExampleAlgorithm_1(_Analysis):
    """
    Example for an analysis class implementing algorithm_1.
    Compute some data and provide a plot and histogram with secondary data (e.g. from fitting plot or histogram).

    This is a specialized analysis class implementing an example analysis routine. For illustrating the analysis
    procedure it only takes a LocData object, creates some random data as result and
    provides plots and a report of the results.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    kwargs
        Parameter that are passed to the algorithm.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata | None
        Metadata about the current analysis routine.
    results : numpy.ndarray, pandas.DataFrame
        Computed results.
    """

    count: int = 0

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        limits: tuple[int, int] = (0, 10),
    ) -> None:
        super().__init__(meta=meta, limits=limits)

    def compute(self, locdata: LocData | None) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData | None
          Localization data that might be clustered.

        Returns
        -------
        Self
        """
        data = None  # take certain elements from locdata
        self.results = _algorithm_1(
            data=data, **self.parameter
        )  # some complicated algorithm
        return self

    def plot(self, ax: mpl.axes.Axes | None = None) -> mpl.axes.Axes:
        return plot(self=self, ax=ax)

    def plot_2(
        self,
        ax: mpl.axes.Axes | None = None,
        bins: str = "auto",
        density: bool = True,
        log: bool = False,
        fit: bool = True,
    ) -> mpl.axes.Axes:
        return plot_2(self=self, ax=ax, bins=bins, density=density, log=log, fit=fit)

    def plot_histogram_fit(self, ax: mpl.axes.Axes | None = None) -> mpl.axes.Axes:
        return plot_histogram_fit(self=self, ax=ax)

    def report(self, path: str | os.PathLike[Any] | None = None) -> mpl.axes.Axes:
        return report(self=self, path=path)


class AnalysisExampleAlgorithm_2(_Analysis):
    """
    Example for an analysis class implementing algorithm_2.
    Compute some data and provide a plot and histogram with secondary data (e.g. from fitting plot or histogram).

    This is a specialized analysis class implementing an example analysis routine. For illustrating the analysis
    procedure it only takes a LocData object, creates some random data as result and
    provides plots and a report of the results.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    kwargs
        Parameter that are passed to the algorithm.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata | None
        Metadata about the current analysis routine.
    results : numpy.ndarray, pandas.DataFrame
        Computed results.
    """

    count = 0

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        n_sample: int = 100,
        seed: RandomGeneratorSeed = None,
    ) -> None:
        super().__init__(meta=meta, n_sample=n_sample, seed=seed)

    def compute(self, locdata: LocData) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
          Localization data that might be clustered.

        Returns
        -------
        Self
          Returns the Analysis class object (self).
        """
        data = locdata  # take certain elements from locdata
        self.results = _algorithm_2(
            data=data, **self.parameter
        )  # some complicated algorithm
        return self

    def plot(
        self, ax: mpl.axes.Axes | npt.NDArray[mpl.axes.Axes] | None = None
    ) -> mpl.axes.Axes | npt.NDArray[mpl.axes.Axes]:
        return plot(self, ax)

    def plot_2(
        self,
        ax: mpl.axes.Axes | None = None,
        bins: str = "auto",
        density: bool = True,
        log: bool = False,
        fit: bool = True,
    ) -> mpl.axes.Axes:
        return plot_2(self=self, ax=ax, bins=bins, density=density, log=log, fit=fit)

    def plot_histogram_fit(
        self, ax: mpl.axes.Axes | npt.NDArray[mpl.axes.Axes] | None = None
    ) -> mpl.axes.Axes | npt.NDArray[mpl.axes.Axes]:
        return plot_histogram_fit(self, ax)

    def report(
        self, path: str | os.PathLike[Any] | None = None
    ) -> mpl.axes.Axes | npt.NDArray[mpl.axes.Axes]:
        return report(self, path)


#
# Interface functions
#
# Now we have a class structure with results that can be further processed.
# Secondary results from e.g. fit procedures are added to the analysis class as new attributes.
#


def plot(self: Any, ax: mpl.axes.Axes | None = None) -> mpl.axes.Axes:
    """
    A specialized plot to give a standardized visualization of results.
    """
    if ax is None:
        ax = plt.gca()

    ax.plot(self.results)
    ax.set(title="Normal Data", xlabel=property, ylabel="PDF")

    return ax


def plot_2(
    self: Any,
    ax: mpl.axes.Axes | npt.NDArray[mpl.axes.Axes] | None = None,
    bins: str = "auto",
    density: bool = True,
    log: bool = False,
    fit: bool = True,
) -> mpl.axes.Axes | npt.NDArray[mpl.axes.Axes]:
    """
    A specialized plot to give a standardized visualization of results - in this case a histogram of results.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 3))
        ax = fig.subplots(nrows=1, ncols=2)
        plt.subplots_adjust(wspace=0)

    # create histogram on first axes
    hist, bins, _ = ax[0].hist(  # type: ignore[index]
        self.results.values,
        bins=bins,
        density=density,
        log=log,
        label=list(self.results),
    )
    ax[0].set(title="Normal Data", xlabel="property", ylabel="PDF")  # type: ignore[index]

    # create legend and results text on second axes
    h_, l_ = ax[0].get_legend_handles_labels()  # type: ignore[index]
    ax[1].legend(  # type: ignore[index]
        h_,
        l_,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        title="Legend",
        frameon=False,
        borderaxespad=0,
    )

    ax[1].set_axis_off()  # type: ignore[index]

    # fit distributions
    if fit:
        plot_histogram_fit(self, ax=ax)

    return ax


def plot_histogram_fit(
    self: Any, ax: mpl.axes.Axes | npt.NDArray[mpl.axes.Axes] | None = None
) -> mpl.axes.Axes | npt.NDArray[mpl.axes.Axes]:
    """
    A specialized plot to give a standardized visualization of results -
    in this case a histogram of results.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 3))
        ax = fig.subplots(nrows=1, ncols=2)
        plt.subplots_adjust(wspace=0)

    # fit distributions
    loc, scale = fit_histogram(self, data=self.results["a"].values, id_="a")

    # plot fit
    x_values = np.linspace(
        stats.norm.ppf(0.01, loc=loc, scale=scale),
        stats.norm.ppf(0.99, loc=loc, scale=scale),
        100,
    )
    ax[0].plot(  # type: ignore[index]
        x_values,
        stats.norm.pdf(x_values, loc=loc, scale=scale),
        "r-",
        lw=3,
        alpha=0.6,
        label="norm pdf",
    )

    # present fit results
    ax[1].text(0, 0.5, "Fit Results:")  # type: ignore[index]
    ax[1].text(  # type: ignore[index]
        0,
        0.5,
        "center: " + str(loc) + "\n" + "sigma: " + str(scale),
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax[1].transAxes,  # type: ignore[index]
        clip_on=False,
    )

    ax[1].set_axis_off()  # type: ignore[index]

    return ax


def fit_histogram(self: Any, data: Any, id_: str) -> tuple[float, float]:
    # MLE fit of distribution on data
    loc, scale = stats.norm.fit(data)

    attribute_center = id_ + "_center"
    attribute_sigma = id_ + "_sigma"
    setattr(self, attribute_center, loc)
    setattr(self, attribute_sigma, scale)

    return loc, scale


# there will be other specific visualization methods for other analysis routines.


def report(
    self: Any, path: str | os.PathLike[Any] | None = None
) -> mpl.axes.Axes | npt.NDArray[mpl.axes.Axes]:
    """
    Provide a report that is either displayed or saved as pdf.
    The report is a figure summarizing all visual representations.
    It is arranged specifically for a particular
    analysis routine.
    """
    fig = plt.figure(figsize=(8.3, 11.7))
    ax: mpl.axes.Axes = fig.subplots(nrows=3, ncols=2)

    # provide the axes elements (i.e. the plots)
    self.plot(ax=ax[0][0])  # type: ignore[index]
    self.plot_2(ax=ax[1][0:2])  # type: ignore[index]

    # adjust figure layout
    plt.tight_layout()

    # save figure as pdf
    if path is not None:
        plt.savefig(
            fname=path,
            dpi=None,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            papertype=None,
            format=None,
            transparent=False,
            bbox_inches=None,
            pad_inches=0.1,
            frameon=None,
        )

    return ax
