"""

Analyze grouped property expectations.

As an example, assume a collection of locdatas representing localization
clusters.
Some property of locdata `loc_property`(i.e. cluster property; e.g. intensity)
might depend on another property `other_loc_property` (e.g. number of
localizations in cluster).
For inspection, all values of loc_property are grouped by other_loc_property
and mean with standard deviations are displayed.

See Also
--------
position_variance_expectation
convex_hull_expectation
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import boost_histogram as bh
import matplotlib.pyplot as plt
import pandas as pd

from locan.analysis.analysis_base import _Analysis
from locan.data.aggregate import Bins

if TYPE_CHECKING:
    import matplotlib as mpl

__all__ = ["GroupedPropertyExpectation"]

logger = logging.getLogger(__name__)


class Collection(Protocol):
    data: pd.DataFrame
    references: Iterable

    def __len__(self):
        pass


@dataclass(repr=False)
class GroupedPropertyExpectationResults:
    values: pd.DataFrame = field(default_factory=pd.DataFrame)
    # with index being reference_index
    # with columns: loc_property, other_loc_property, value_to_expectation_ratio
    grouped: pd.DataFrame = field(default_factory=pd.DataFrame)
    # with index being other_loc_property
    # with columns: loc_property_mean, loc_property_std, expectation


class GroupedPropertyExpectation(_Analysis):
    """
    Analyze variation of localization property
    in relation to other localization property that is grouped.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    loc_property: str
        The localization property to analyze.
    other_loc_property: str
        The localization property to group on.
    expectation : int | float | Mapping | pd.Series | None
        The expected value for all or each other localization property.

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : GroupedPropertyExpectationResults
        Computed results.
    distribution_statistics : Distribution_stats, None
        Distribution parameters derived from MLE fitting of results.grouped.
    """

    def __init__(
        self, meta=None, loc_property=None, other_loc_property=None, expectation=None
    ):
        super().__init__(
            meta=meta,
            loc_property=loc_property,
            other_loc_property=other_loc_property,
            expectation=expectation,
        )
        self.expectation = expectation
        self.results = None
        self.distribution_statistics = None

    def compute(self, locdata=None):
        """
        Run the computation.

        Parameters
        ----------
        locdata : Collection
            Localization data.

        Returns
        -------
        Analysis class
            Returns the Analysis class object (self).
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        loc_property = self.parameter["loc_property"]
        other_loc_property = self.parameter["other_loc_property"]

        self.results = GroupedPropertyExpectationResults()
        self.results.values = locdata.data.loc[:, [loc_property, other_loc_property]]

        grouped = self.results.values.groupby(other_loc_property)

        self.results.grouped[loc_property + "_mean"] = grouped.mean()
        self.results.grouped[loc_property + "_std"] = grouped.std()

        if self.expectation is None:
            self.results.grouped["expectation"] = pd.NA
        elif isinstance(self.expectation, (int, float)):
            self.results.grouped["expectation"] = self.expectation
        elif isinstance(self.expectation, pd.Series):
            indices = self.results.grouped.index
            self.results.grouped["expectation"] = self.expectation.loc[indices]
        elif isinstance(self.expectation, Mapping):
            self.results.grouped["expectation"] = [
                self.expectation[index_] for index_ in self.results.grouped.index
            ]

        self.results.values["expectation"] = self.results.grouped.loc[
            self.results.values[other_loc_property], "expectation"
        ].to_numpy()

        self.results.values["value_to_expectation_ratio"] = (
            self.results.values[loc_property] / self.results.values["expectation"]
        )

        return self

    def plot(self, ax=None, **kwargs) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        variances as function of localization counts.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which to show the image
        kwargs : dict
            Other parameters passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        self.results.values.plot(
            kind="scatter",
            alpha=0.2,
            x=self.parameter["other_loc_property"],
            y=self.parameter["loc_property"],
            color="gray",
            ax=ax,
            **dict(dict(legend=False), **kwargs),
        )

        self.results.grouped.plot(
            kind="line",
            marker=".",
            x=None,
            y=self.parameter["loc_property"] + "_mean",
            color="blue",
            ax=ax,
        )

        if not self.results.grouped.expectation.isna().all():
            self.results.grouped.plot(
                kind="line",
                x=None,
                y="expectation",
                color="black",
                ax=ax,
            )

        ax.set(
            title="Grouped Property Expectation",
            xlabel=self.parameter["other_loc_property"],
            ylabel=self.parameter["loc_property"],
        )

        return ax

    def hist(
        self,
        ax=None,
        bins=None,
        n_bins=None,
        bin_size=None,
        bin_edges=None,
        bin_range=None,
        log=True,
        fit=False,
        **kwargs,
    ) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        2-dimensional histogram.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which to show the image
        bins : Bins | boost_histogram.axis.Axis | boost_histogram.axis.AxesTuple | None
            The bin specification as defined in :class:`Bins`
        bin_edges : Sequence[float] | Sequence[Sequence[float]] | None
            Bin edges for all or each dimension
            with shape (dimension, n_bin_edges).
        bin_range : tuple[float, float] | Sequence[float] | Sequence[Sequence[float]]
            Minimum and maximum edge for all or each dimensions
            with shape (2,) or (dimension, 2).
        n_bins : int | Sequence[int] | None
            The number of bins for all or each dimension.
            5 yields 5 bins in all dimensions.
            (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
        bin_size : float | Sequence[float] | Sequence[Sequence[float]] | None
            The size of bins for all or each bin and for all or each dimension
            with shape (dimension,) or (dimension, n_bins).
            5 would describe bin_size of 5 for all bins in all dimensions.
            ((2, 5),) yield bins of size (2, 5) for one dimension.
            (2, 5) yields bins of size 2 for one dimension and 5 for the other
            dimension.
            ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and
            (1, 3) for the other dimension.
            To specify arbitrary sequence of `bin_size` use `bin_edges` instead.
        log : bool
            Flag for plotting on a log scale.
        kwargs : dict
            Other parameters passed to :func:`matplotlib.pyplot.pcolormesh`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        fig = ax.get_figure()

        other_loc_property = self.parameter["other_loc_property"]
        loc_property = self.parameter["loc_property"]

        if all(
            item_ is None for item_ in [bins, n_bins, bin_size, bin_edges, bin_range]
        ):
            max_other_loc_property_ = self.results.grouped.index.max()
            max_loc_property_ = self.results.values[loc_property].max()

            if log:
                axes = bh.axis.AxesTuple(
                    (
                        bh.axis.Regular(
                            max_other_loc_property_,
                            1,
                            max_other_loc_property_,
                            transform=bh.axis.transform.log,
                        ),
                        bh.axis.Regular(
                            50, 1, max_loc_property_, transform=bh.axis.transform.log
                        ),
                    )
                )
            else:
                axes = bh.axis.AxesTuple(
                    (
                        bh.axis.Regular(
                            max_other_loc_property_, 1, max_other_loc_property_
                        ),
                        bh.axis.Regular(50, 1, max_loc_property_),
                    )
                )
            bins = Bins(bins=axes)
        else:
            try:
                bins = Bins(
                    bins,
                    n_bins,
                    bin_size,
                    bin_edges,
                    bin_range,
                    labels=[loc_property, other_loc_property],
                )
            except ValueError as exc:
                # todo: check if message is appropriate
                # the error is raised again only to adapt the message.
                raise ValueError(
                    "Bin dimension and len of `loc_properties` is incompatible."
                ) from exc
        axes = bins.boost_histogram_axes
        histogram = bh.Histogram(*axes)
        histogram.reset()
        histogram.fill(
            self.results.values[other_loc_property], self.results.values[loc_property]
        )
        mesh = ax.pcolormesh(*histogram.axes.edges.T, histogram.view().T, **kwargs)
        fig.colorbar(mesh)

        self.results.grouped.plot(
            kind="line",
            marker=".",
            x=None,
            y=self.parameter["loc_property"] + "_mean",
            color="blue",
            ax=ax,
        )

        if not self.results.grouped.expectation.isna().all():
            self.results.grouped.plot(
                kind="line",
                x=None,
                y="expectation",
                color="white",
                ax=ax,
            )

        ax.set(
            title="Grouped Property Expectation",
            xlabel=self.parameter["other_loc_property"],
            ylabel=self.parameter["loc_property"],
        )

        if log:
            ax.set(xscale="log", yscale="log")

        if fit:
            raise NotImplementedError

        return ax
