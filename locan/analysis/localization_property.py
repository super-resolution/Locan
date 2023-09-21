"""

Analyze localization property.

Localizations come with a range of properties including position coordinates,
emission strength, local background etc..
Most properties represent random variables that were drawn from an unknown
probability distribution.
It is often useful to analyze the properties from all localizations within a
selection and estimate the corresponding probability distribution.

"""
from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

if TYPE_CHECKING:
    import matplotlib as mpl
    import pandas as pd

    from locan.data.locdata import LocData

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis, _list_parameters

__all__: list[str] = ["LocalizationProperty"]

logger = logging.getLogger(__name__)


# The algorithms


def _localization_property(
    locdata: LocData, loc_property: str = "intensity", index: str | None = None
) -> pd.DataFrame:
    if index is None:
        results = locdata.data[[loc_property]]
    else:
        results = locdata.data[[loc_property, index]].set_index(index)
        results = results.sort_index()

    return results


# The specific analysis classes


class LocalizationProperty(_Analysis):
    """
    Analyze localization property with respect to probability density or
    variation over a specified index.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    loc_property : str
        The property to analyze.
    index : str | None
        The property name that should serve as index (i.e. x-axis in x-y-plot)

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    parameter : dict[str, Any]
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.DataFrame
        Computed results.
    distribution_statistics : Distribution_stats | None
        Distribution parameters derived from MLE fitting of results.
    """

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        loc_property: str = "intensity",
        index: str | None = None,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)

        self.results: pd.DataFrame | None = None
        self.distribution_statistics: _DistributionFits | None = None

    def compute(self, locdata: LocData) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata
            Localization data.

        Returns
        -------
        Self
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        self.results = _localization_property(locdata=locdata, **self.parameter)
        return self

    def fit_distributions(
        self,
        distribution: str | stats.rv_continuous = stats.expon,
        with_constraints: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Fit probability density functions to the distributions of
        `loc_property` values in the results
        using MLE (scipy.stats).

        If with_constraints is true we put the following constraints on the
        fit procedure:
        If distribution is expon then
        `floc=np.min(self.analysis_class.results[self.loc_property].values)`.

        Parameters
        ----------
        distribution
            Distribution model to fit.
        with_constraints
            Flag to use predefined constraints on fit parameters.
        kwargs
            Other parameters are passed to `scipy.stat.distribution.fit()`.
        """
        if self:
            self.distribution_statistics = _DistributionFits(self)
            self.distribution_statistics.fit(
                distribution, with_constraints=with_constraints, **kwargs
            )
        else:
            logger.warning("No results available to fit.")

    def plot(
        self, ax: mpl.axes.Axes | None = None, window: int = 1, **kwargs: Any
    ) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        running average of results over window size.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to show the image
        window
            Window for running average that is applied before plotting.
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
            return ax

        self.results.rolling(window=window, center=True).mean().plot(
            ax=ax, **dict(dict(legend=False), **kwargs)
        )
        ax.set(
            title=f"{self.parameter['loc_property']}({self.parameter['index']})\n (window={window})",
            xlabel=self.parameter["index"],
            ylabel=self.parameter["loc_property"],
        )

        return ax

    def hist(
        self,
        ax: mpl.axes.Axes | None = None,
        bins: int | Sequence[int | float] | str = "auto",
        log: bool = True,
        fit: bool | None = True,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing
        hist(results). Nan entries are ignored.

        Parameters
        ----------
        ax
            The axes on which to show the image
        bins
            Bin specifications (passed to :func:`matplotlib.hist`).
        log
            Flag for plotting on a log scale.
        fit
            Flag indicating if distribution fit is shown.
            The fit will only be computed if `distribution_statistics` is None.
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.hist`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
            return ax

        ax.hist(
            self.results.dropna(axis=0).values,
            bins=bins,
            **dict(dict(density=True, log=log), **kwargs),
        )
        ax.set(
            title=self.parameter["loc_property"],
            xlabel=self.parameter["loc_property"],
            ylabel="PDF",
        )

        # fit distributions:
        if fit:
            if self.distribution_statistics is None:
                self.fit_distributions()

            assert (  # type narrowing # noqa: S101
                self.distribution_statistics is not None
            )
            self.distribution_statistics.plot(ax=ax)

        return ax


# todo add Dependence_stats to fit a plot to a linear function, log function, or exponential decay.


class _DistributionFits:
    """
    Handle for distribution fits.

    This class is typically instantiated by LocalizationProperty methods.
    It holds the statistical parameters derived by fitting the result
    distributions using MLE (scipy.stats).
    Statistical parameters are defined as described in
    :ref:(https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html)

    Parameters
    ----------
    analyis_class : LocalizationPrecision
        The analysis class with result data to fit.

    Attributes
    ----------
    analyis_class : LocalizationProperty
        The analysis class with result data to fit.
    loc_property : str
        The LocData property for which to fit an appropriate distribution.
    distribution : scipy.stats.rv_continuous | None
        Distribution model to fit.
    parameters : list[str]
        Distribution parameters.
    """

    def __init__(self, analysis_class: LocalizationProperty) -> None:
        self.analysis_class = analysis_class
        self.loc_property = self.analysis_class.parameter["loc_property"]
        self.distribution: stats.rv_continuous | None = None
        self.parameters: list[str] = []

    def fit(
        self,
        distribution: stats.rv_continuous,
        with_constraints: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Fit scipy.stats.rv_continuous to analysis_class.results[loc_property].

        If with_constraints is true we put the following constraints on the
        fit procedure:
        If distribution is expon then
        floc=np.min(self.analysis_class.results[self.loc_property].values).

        Parameters
        ----------
        distribution
            Distribution model to fit.
        with_constraints
            Flag to use predefined constraints on fit parameters.
        kwargs
            Other parameters are passed to `scipy.stats.rv_continuous.fit()`.
        """
        if self.analysis_class.results is None:
            logger.warning("No results available to fit.")
            return None
        self.distribution = distribution
        for param in _list_parameters(distribution):
            self.parameters.append(self.loc_property + "_" + param)

        if with_constraints and self.distribution == stats.expon:
            # MLE fit of exponential distribution with constraints
            fit_results = stats.expon.fit(
                self.analysis_class.results[self.loc_property].values,
                **dict(
                    dict(
                        floc=np.min(
                            self.analysis_class.results[self.loc_property].values
                        )
                    ),
                    **kwargs,
                ),
            )
            for parameter, result in zip(self.parameters, fit_results):
                setattr(self, parameter, result)
        else:
            fit_results = self.distribution.fit(
                self.analysis_class.results[self.loc_property].values, **kwargs
            )
            for parameter, result in zip(self.parameters, fit_results):
                setattr(self, parameter, result)

    def plot(self, ax: mpl.axes.Axes | None = None, **kwargs: Any) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        probability distribution functions of fitted results.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to show the image.
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.distribution is None:
            return ax

        # plot fit curve
        parameter = self.parameter_dict().values()
        x_values = np.linspace(
            self.distribution.ppf(0.001, *parameter),
            self.distribution.ppf(0.999, *parameter),
            100,
        )
        ax.plot(
            x_values,
            self.distribution.pdf(x_values, *parameter),
            "r-",
            **dict(
                dict(lw=3, alpha=0.6, label=str(self.distribution.name) + " pdf"),
                **kwargs,
            ),
        )

        return ax

    def parameter_dict(self) -> dict[str, float]:
        """Dictionary of fitted parameters."""
        return {k: self.__dict__[k] for k in self.parameters}
