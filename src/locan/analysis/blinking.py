"""
Compute on- and off-periods from localization frames.

Assuming that the provided localizations are acquired from the same label,
we analyze the times of recording as
provided by the `frame` property.
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Sequence
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis, _list_parameters
from locan.data.locdata import LocData

__all__: list[str] = ["BlinkStatistics"]

logger = logging.getLogger(__name__)


# The algorithms


def _blink_statistics(
    locdata: LocData | npt.ArrayLike,
    memory: int = 0,
    remove_heading_off_periods: bool = True,
) -> dict[str, npt.NDArray[np.int_ | np.float_] | list[int | float]]:
    """
    Estimate on and off times from the frame values provided.

    On and off-periods and the first frame of each period are determined from
    the sorted frame values.
    A series of frame values that constantly increase by one is considered an
    on-period.
    Each series of missing frame values between two given frame values is
    considered an off-period.

    Parameters
    ----------
    locdata
        Localization data or just the frame values of given localizations.
    memory
        The maximum number of intermittent frames without any localization
        that are still considered to belong to the same on-period.
    remove_heading_off_periods
        Flag to indicate if off-periods at the beginning of the series are
        excluded.

    Returns
    -------
    dict[str, npt.NDArray[int | float] | list[int | float]]
        'on_periods' and 'off_periods' in units of frame numbers.
        'on_periods_frame' and 'off_periods_frame' with the first frame in
        each on/off-period.
        'on_periods_indices' are groups of indices to the input frames or
        more precise np.unique(frames)
    """
    if isinstance(locdata, LocData):
        frames = locdata.data.frame.values
    else:
        frames = locdata  # type: ignore[assignment]

    frames, counts = np.unique(frames, return_counts=True)  # type: ignore[call-overload]

    # provide warning if duplicate frames are found. This should not be the case for appropriate localization clusters.
    if np.any(counts > 1):
        counts_larger_one = counts[counts > 1]
        logger.warning(
            f"There are {sum(counts_larger_one) - len(counts_larger_one)} "
            f"duplicated frames found that will be ignored."
        )

    # shift frames and add first frame if no zero frame present to account for initial off_period
    first_frame = frames[0]
    frames_ = np.insert(frames + 1, 0, 0)  # type: ignore[operator]

    differences = np.insert(
        np.diff(frames_), 0, 1
    )  # first frame should be one since zero frame now is always on.

    # on_ and off_periods
    mask = np.where(differences > memory + 1, True, False)
    off_periods = differences[mask] - 1
    on_periods_frame = np.insert(frames_[mask], 0, 0)
    off_periods_frame = frames_[mask] - off_periods

    indices_on = np.nonzero(mask)[0]
    # the following paragraph corresponds to this python equivalent:
    #
    # groups = np.split(differences, indices_on)
    # the sum is taken to include memory > 0.
    # one is added since a single localization is considered to be on for one frame.
    # on_periods = np.array([np.sum(group[1:]) + 1 for group in groups])
    if indices_on.size:
        differences[0] = 1
        differences[indices_on] = 1
        cumsum = differences.cumsum()
        first = cumsum[indices_on[0] - 1]
        middle = cumsum[indices_on - 1][1:] - cumsum[indices_on - 1][:-1]
        last = cumsum[-1] - cumsum[indices_on - 1][-1]
        on_periods = np.insert(middle, 0, first)
        on_periods = np.append(on_periods, last)
    else:
        on_periods = np.array([frames_[-1] + 1])

    # grouped indices to all localizations in each on-period
    indices = np.arange(-1, len(frames_) - 1)
    on_periods_indices = np.split(indices, indices_on)

    # clean up initial shift and insert
    if first_frame == 0:  # the first frame is on
        on_periods[0] = on_periods[0] - 1
        on_periods_frame[0] = on_periods_frame[0] + 1
        on_periods_indices[0] = on_periods_indices[0][1:]

    elif first_frame > memory:  # there is an initial off_period
        on_periods = on_periods[1:]
        on_periods_frame = on_periods_frame[1:]
        if remove_heading_off_periods:
            off_periods = off_periods[1:]
            off_periods_frame = off_periods_frame[1:]
        on_periods_indices = on_periods_indices[1:]

    elif (
        first_frame <= memory
    ):  # there is an initial off_period integrated in the first on_period
        if remove_heading_off_periods:
            on_periods[0] = on_periods[0] - first_frame - 1
            on_periods_frame[0] = first_frame + 1
        else:
            on_periods[0] = on_periods[0] - 1
            on_periods_frame[0] = on_periods_frame[0] + 1
        on_periods_indices[0] = on_periods_indices[0][1:]

    on_periods_frame = on_periods_frame - 1
    off_periods_frame = off_periods_frame - 1

    return dict(
        on_periods=on_periods,
        on_periods_frame=on_periods_frame,
        off_periods=off_periods,
        off_periods_frame=off_periods_frame,
        on_periods_indices=on_periods_indices,
    )


# The specific analysis classes


class BlinkStatistics(_Analysis):
    """
    Estimate on and off times from the frame values provided.

    On and off-periods and the first frame of each period are determined from
    the sorted frame values.
    A series of frame values that constantly increase by one is considered an
    on-period.
    Each series of missing frame values between two given frame values is
    considered an off-period.

    A log warning is provided if a frame number occurs multiple times.

    Missing localizations within an on-period can be taken into account by
    increasing the `memory` parameter.
    There is no way to correct for false positive localizations.

    Parameters
    ----------
    memory : int
        The maximum number of intermittent frames without any localization
        that are still considered to belong to the same on-period.
    remove_heading_off_periods : bool
        Flag to indicate if off-periods at the beginning of the series are
        excluded.
    meta : metadata_analysis_pb2.AMetadata | None
        Metadata about the current analysis routine.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict[str, Any]
        A dictionary with all settings for the current computation.
    meta : metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : dict[str, npt.NDArray[np.int_ | np.float_] | list[int | float]] | None
        'on_periods' and 'off_periods' in units of frame numbers.
        'on_periods_frame' and 'off_periods_frame' with the first frame in
        each on/off-period.
        'on_periods_indices' are groups of indices to the input frames or more
         precise np.unique(frames)
    distribution_statistics : dict[str, Any]
        Distribution parameters derived from MLE fitting of results.
    """

    count = 0

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        memory: int = 0,
        remove_heading_off_periods: bool = True,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)

        self.results: dict[
            str, npt.NDArray[np.int_ | np.float_] | list[int | float]
        ] | None = None
        self.distribution_statistics: dict[str, Any] = {}

    def compute(self, locdata: LocData | npt.ArrayLike) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata
            Localization data or just the frame values of given localizations.

        Returns
        -------
        Self
        """
        if not len(locdata):  # type: ignore
            logger.warning("Locdata is empty.")
            return self

        self.results = _blink_statistics(locdata=locdata, **self.parameter)
        return self

    def fit_distributions(
        self,
        distribution: str | stats.rv_continuous = stats.expon,
        data_identifier: str | Iterable[str] = ("on_periods", "off_periods"),
        with_constraints: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Fit probability density functions to the distributions of on- and
        off-periods in the results using MLE (scipy.stats).

        If with_constraints is true we put the following constraints on the
        fit procedure:
        If distribution is expon then
        floc=np.min(self.analysis_class.results[self.loc_property].values).

        Parameters
        ----------
        distribution
            Distribution model to fit.
        data_identifier
            String to identify the data in `results` for which to fit an
            appropriate distribution, here
            'on_periods' or 'off_periods'. For True all are fitted.
        with_constraints
            Flag to use predefined constraints on fit parameters.
        kwargs
            Other parameters are passed to the `scipy.stat.distribution.fit()`
            function.

        Returns
        -------
        None
        """
        if self.results is None:
            logger.warning("No results available to fit.")
        else:
            if isinstance(data_identifier, (tuple, list)):
                data_identifier_ = data_identifier
            else:
                data_identifier_ = (data_identifier,)

            for data_id in data_identifier_:
                self.distribution_statistics[data_id] = _DistributionFits(
                    self, data_identifier=data_id, distribution=distribution
                )
                self.distribution_statistics[data_id].fit(
                    with_constraints=with_constraints, **kwargs
                )

    def hist(
        self,
        data_identifier: str = "on_periods",
        ax: mpl.axes.Axes | None = None,
        bins: int | Sequence[int | float] | str = "auto",
        log: bool = True,
        fit: bool | None = True,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing
        hist(results).

        Parameters
        ----------
        data_identifier
            'on_periods' or 'off_periods'.
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
            self.results[data_identifier],
            bins=bins,
            **dict(dict(density=True, log=log), **kwargs),
        )
        ax.set(
            title=f"Distribution of {data_identifier}",
            xlabel=f"{data_identifier} (frames)",
            ylabel="PDF",
        )

        # fit distributions:
        if fit:
            if data_identifier in self.distribution_statistics and isinstance(
                self.distribution_statistics[data_identifier], _DistributionFits
            ):
                self.distribution_statistics[data_identifier].plot(ax=ax)
            else:
                self.fit_distributions(data_identifier=data_identifier)
                self.distribution_statistics[data_identifier].plot(ax=ax)

        return ax


class _DistributionFits:
    """
    Handle for distribution fits.

    This class is typically instantiated by specific Analysis methods.
    It holds the statistical parameters derived by fitting the result
    distributions using MLE (scipy.stats).
    Statistical parameters are defined as described in
    :ref:(https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html)

    Parameters
    ----------
    analysis_class
        The analysis class with result data to fit.
    distribution
        Distribution model to fit.
    data_identifier
        String to identify the data in `results` for which to fit an
        appropriate distribution

    Attributes
    ----------
    analysis_class : _Analysis
        The analysis class with result data to fit.
    distribution : scipy.stats.rv_continuous
        Distribution model to fit.
    data_identifier : str
        String to identify the data in `results` for which to fit an
        appropriate distribution
    parameters : list[str]
        Distribution parameters.
    """

    def __init__(
        self,
        analysis_class: _Analysis,
        distribution: str | stats.rv_continuous,
        data_identifier: str,
    ) -> None:
        self.analysis_class: _Analysis = analysis_class
        self.distribution: stats.rv_continuous = distribution
        self.data_identifier: str = data_identifier
        self.parameters: list[str] = []

    def __repr__(self) -> str:
        """Return representation of the _DistributionFits class."""
        param_dict = dict(
            analysis_class=self.analysis_class.__class__.__name__,
            distribution=self.distribution.__class__.__name__,
            data_identifier=self.data_identifier,
        )
        param_string = ", ".join((f"{key}={val}" for key, val in param_dict.items()))
        return f"{self.__class__.__name__}({param_string})"

    def fit(self, with_constraints: bool = True, **kwargs: Any) -> None:
        """
        Fit scipy.stats.rv_continuous to analysis_class.results[data_identifier].

        If with_constraints is true we put the following constraints on the
        fit procedure:
        If distribution is expon then
        floc=np.min(self.analysis_class.results[self.data_identifier].values).

        Parameters
        ----------
        with_constraints
            Flag to use predefined constraints on fit parameters.
        kwargs
            Other parameters are passed to the `scipy.stat.distribution.fit()`
            function.
        """
        if self.analysis_class.results is None:
            raise ValueError("Compute results before fitting.")

        # set data
        if isinstance(self.analysis_class.results, pd.DataFrame):
            data = self.analysis_class.results[self.data_identifier].values
        else:
            data = self.analysis_class.results[self.data_identifier]

        # define parameter names
        for param in _list_parameters(self.distribution):
            self.parameters.append(self.data_identifier + "_" + param)

        # perform fit
        if with_constraints and self.distribution == stats.expon:
            # MLE fit of exponential distribution with constraints
            fit_results = stats.expon.fit(
                data, **dict(dict(floc=np.min(data)), **kwargs)  # type: ignore[arg-type]
            )
            for parameter, result in zip(self.parameters, fit_results):
                setattr(self, parameter, result)
        else:
            fit_results = self.distribution.fit(data, **kwargs)
            for parameter, result in zip(self.parameters, fit_results):
                setattr(self, parameter, result)

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
        mpl.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.distribution is None:
            return ax

        # plot fit curve
        parameter = self.parameter_dict().values()
        x_values = np.linspace(
            self.distribution.ppf(1e-4, *parameter),
            self.distribution.ppf(1 - 1e-4, *parameter),
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

    def parameter_dict(self) -> dict[str, int | float]:
        """Dictionary of fitted parameters."""
        return {k: self.__dict__[k] for k in self.parameters}
