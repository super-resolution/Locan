"""

Compute localizations per frame.

"""
from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    import matplotlib as mpl

    from locan.data.locdata import LocData

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis

__all__: list[str] = ["LocalizationsPerFrame"]

logger = logging.getLogger(__name__)


# The algorithms


def _localizations_per_frame(
    locdata: LocData | npt.ArrayLike,
    norm: int | float | str | None = None,
    time_delta: int | float | str | pd.Timedelta | None = "integration_time",
    resample: pd.DateOffset | pd.Timedelta | str | None = None,
    **kwargs: Any,
) -> pd.Series[Any]:
    """
    Compute localizations per frame.

    Parameters
    ----------
    locdata
       Points in time: either localization data that contains a column `frame`
       or an array with time points.
    norm
        Normalization factor that can be None, a number, or another property in
         `locdata`.
    time_delta
        Time per frame in milliseconds. String must specify the unit like
        "10ms".
         For "integration_time" the time is taken from
        locdata.meta.experiment.setups[0].optical_units[0].detection.camera.integration_time
    resample
        Parameter for :func:`pandas.Series.resample`: The offset string or
        object representing target conversion.
    kwargs
        Other parameters passed to :func:`pandas.Series.resample`.


    Returns:
    --------
    pandas.Series
    """
    # normalization
    if norm is None:
        normalization_factor: int | float = 1
        series_name = "n_localizations"
    elif isinstance(norm, str):
        try:
            normalization_factor = locdata.properties[norm]  # type: ignore
            series_name = "n_localizations / " + norm
        except (AttributeError, KeyError) as exception:
            raise KeyError(
                "normalization must be a valid property name in locdata.properties."
            ) from exception
    elif isinstance(norm, (int, float)):
        normalization_factor = norm
        series_name = f"n_localizations / {norm}"
    else:
        raise TypeError(
            "normalization should be None, a number or a valid property name."
        )

    try:
        frames_ = locdata.data.frame.astype(int)  # type: ignore
    except AttributeError:
        frames_ = np.asarray(locdata)

    frames, frame_counts = np.unique(frames_, return_counts=True)
    series = pd.Series(frame_counts, index=frames, dtype=float) / normalization_factor
    series.index.name = "frame"
    series.name = series_name

    if time_delta is None:
        pass
    elif isinstance(time_delta, pd.Timedelta):
        series.index = frames * time_delta
        series.index.name = "time"
    elif isinstance(time_delta, (int, float)):
        series.index = pd.to_timedelta(frames * time_delta, unit="ms")
        series.index.name = "time"
    elif time_delta == "integration_time":
        try:
            time_delta = (
                locdata.meta.experiment.setups[0]  # type: ignore
                .optical_units[0]
                .detection.camera.integration_time.ToTimedelta()
            )
            series.index = frames * time_delta
            series.index.name = "time"
        except (IndexError, AttributeError):
            logger.warning(
                "integration_time not available in locdata.meta - frames used instead."
            )
    elif isinstance(time_delta, str):
        time_delta = pd.Timedelta(time_delta)
        series.index = frames * time_delta
        series.index.name = "time"
    else:
        raise ValueError("The input for time_delta is not implemented.")

    if resample is not None:
        series = series.resample(resample, **kwargs).sum()

    return series


# The specific analysis classes


@dataclass(repr=False)
class _Results:
    time_series: pd.Series[Any]

    def accumulation_time(self, fraction: float = 0.5) -> int:
        normalized_cumulative_time_trace = (
            self.time_series.cumsum() / self.time_series.sum()
        )
        accumulation_time: int = normalized_cumulative_time_trace.gt(fraction).idxmax()
        return accumulation_time


class LocalizationsPerFrame(_Analysis):
    """
    Compute localizations per frame.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    norm : int | float | str | None
        Normalization factor that can be None, a number, or another property
        in `locdata`.
    time_delta : int | float | str | pd.Timedelta | None
        Time per frame in milliseconds. String must specify the unit like
        "10ms".
         For "integration_time" the time is taken from
        locdata.meta.experiment.setups[0].optical_units[0].detection.camera.integration_time
    resample : DateOffset | Timedelta | str
        Parameter for :func:`pandas.Series.resample`: The offset string or
        object representing target conversion.
    kwargs
        Other parameters passed to :func:`pandas.Series.resample`.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.Series
        Computed results.
    distribution_statistics : Distribution_fits object | None
        Distribution parameters derived from MLE fitting of results.
    """

    count = 0

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        norm: int | float | str | None = None,
        time_delta: int | float | str | pd.Timedelta | None = "integration_time",
        resample: pd.DateOffset | pd.Timedelta | str | None = None,
        **kwargs: Any,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.results = None
        self.distribution_statistics: _DistributionFits | None = None

    def compute(self, locdata: LocData | npt.ArrayLike) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata
           Points in time: either localization data that contains a column
           `frame` or an array with time points.

        Returns
        -------
        Self
        """
        if not len(locdata):  # type: ignore
            logger.warning("Locdata is empty.")
            self.distribution_statistics = None
            self.results = None
            return self

        self.distribution_statistics = None
        self.results = _Results(
            time_series=_localizations_per_frame(locdata=locdata, **self.parameter)
        )
        return self

    def fit_distributions(self, **kwargs: Any) -> None:
        # todo: fix kwarg
        """
        Fit probability density functions to the distributions of `
        loc_property` values in the results using MLE (scipy.stats).

        Parameters
        ----------
        loc_property : str
            The LocData property for which to fit an appropriate distribution;
            if None all plots are shown.
        """
        if self:
            self.distribution_statistics = _DistributionFits(self)
            self.distribution_statistics.fit(**kwargs)
        else:
            logger.warning("No results available to fit.")

    def plot(
        self,
        ax: mpl.axes.Axes | None = None,
        window: int = 1,
        cumulative: bool = False,
        normalize: bool = False,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        running average of results over window size.

        Parameters
        ----------
        ax
            The axes on which to show the image
        window
            Window for running average that is applied before plotting.
        cumulative
            Plot the cumulated results if true.
        normalize
            Normalize cumulative plot to the last value
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

        series = self.results.time_series

        # prepare plot
        if cumulative and normalize:
            _results = series.cumsum() / series.sum()
        elif cumulative and not normalize:
            _results = series.cumsum()
        elif not cumulative:
            _results = series
        else:
            return ax

        _results.rolling(window=window, center=True).mean().plot(ax=ax, **kwargs)

        ax.set(
            title=f"Localizations per Frame\n (window={window})",
            xlabel=series.index.name,
            ylabel=f"{series.name} (cumulative)" if cumulative else series.name,
        )

        return ax

    def hist(
        self,
        ax: mpl.axes.Axes | None = None,
        fit: bool = True,
        bins: int | Sequence[int | float] | str = "auto",
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing
        hist(results).

        Parameters
        ----------
        ax
            The axes on which to show the image
        bins
            Bin specifications (passed to matplotlib.hist).
        fit
            Flag indicating if distributions fit are shown.
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

        series = self.results.time_series

        ax.hist(
            series.values, bins=bins, **dict(dict(density=True, log=False), **kwargs)
        )
        ax.set(title="Localizations per Frame", xlabel=series.name, ylabel="PDF")

        if fit:
            if self.distribution_statistics is None:
                self.fit_distributions()

            assert (  # type narrowing # noqa: S101
                self.distribution_statistics is not None
            )
            self.distribution_statistics.plot(ax=ax)

        return ax


# todo: add fit function


class _DistributionFits:
    """
    Handle for distribution fits.

    It holds the statistical parameters derived by fitting the result
    distributions using MLE (scipy.stats).
    Statistical parameters are defined as described in
    :ref:(https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html)

    Parameters
    ----------
    analyis_class : LocalizationsPerFrame
        The analysis class with result data to fit.

    Attributes
    ----------
    analyis_class : LocalizationsPerFrame
        The analysis class with result data to fit.
    loc_property : str | None
        The LocData property for which to fit an appropriate distribution
    distribution : scipy.stats.rv_continuous | None
        Distribution model to fit.
    parameters : list[str]
    """

    def __init__(self, analysis_class: LocalizationsPerFrame) -> None:
        self.analysis_class = analysis_class
        if self.analysis_class.results is None:
            self.loc_property: str | None = None
        else:
            self.loc_property = self.analysis_class.results.time_series.name
        self.distribution: stats.rv_continuous | None = None
        self.parameters: list[str] = []

    def fit(
        self, distribution: stats.rv_continuous = stats.norm, **kwargs: Any
    ) -> None:
        """
        Fit scipy.stats.rv_continuous to analysis_class.results[loc_property].

        If with_constraints is true we put the following constraints on the fit
        procedure:
        If distribution is expon then
        floc=np.min(self.analysis_class.results[self.loc_property].values).

        Parameters
        ----------
        distribution
            Distribution model to fit.
        kwargs
            Other parameters are passed to
            :func:`scipy.stat.distribution.fit()`.
        """
        if self.analysis_class.results is None:
            return
        assert self.loc_property is not None  # type narrowing # noqa: S101

        self.distribution = distribution

        loc, scale = self.distribution.fit(
            self.analysis_class.results.time_series.values, **kwargs
        )
        self.parameters.extend(
            [self.loc_property + "_center", self.loc_property + "_sigma"]
        )
        setattr(self, self.loc_property + "_center", loc)
        setattr(self, self.loc_property + "_sigma", scale)

    def plot(self, ax: mpl.axes.Axes | None = None, **kwargs: Any) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        probability distribution functions of fitted results.

        Parameters
        ----------
        ax
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
        _center, _sigma = self.parameter_dict().values()

        x_values = np.linspace(
            self.distribution.ppf(0.001, loc=_center, scale=_sigma),
            self.distribution.ppf(0.999, loc=_center, scale=_sigma),
            100,
        )
        ax.plot(
            x_values,
            self.distribution.pdf(x_values, loc=_center, scale=_sigma),
            "r-",
            **dict(
                dict(lw=3, alpha=0.6, label=str(self.distribution.name) + " pdf"),
                **kwargs,
            ),
        )
        return ax

    def parameter_dict(self) -> dict[str, float]:
        """Dictionary of fitted parameters."""
        if self.parameters is None:
            return dict()
        else:
            return {k: self.__dict__[k] for k in self.parameters}
