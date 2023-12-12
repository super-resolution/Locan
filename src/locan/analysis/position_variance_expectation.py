"""

Analyze variance of localization coordinates.

Localization coordinates in localization clusters come with a certain variance.
In resolution-limited clusters the variance is determined by the
localization precision.

A close look at the variance of localization coordinates as function of
localization counts helps to characterize localization clusters [1]_.

References
----------
.. [1] Ebert V, Eiring P, Helmerich DA, Seifert R, Sauer M, Doose S.
   Convex hull as diagnostic tool in single-molecule localization microscopy.
   Bioinformatics 38(24), 2022, 5421-5429, doi: 10.1093/bioinformatics/btac700.

"""
# todo add fit procedure to estimate variances
from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import boost_histogram as bh
import matplotlib.pyplot as plt
import pandas as pd

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis
from locan.data.aggregate import Bins
from locan.data.validation import _check_loc_properties
from locan.utils.statistics import biased_variance

if TYPE_CHECKING:
    import matplotlib as mpl

    from locan.data.locdata import LocData

__all__: list[str] = ["PositionVarianceExpectation"]

logger = logging.getLogger(__name__)


# The algorithms


def _property_variances(
    collection: LocData,
    loc_property: str,
    biased: bool = False,
) -> pd.DataFrame:
    assert collection is not None and isinstance(  # type narrowing # noqa: S101
        collection.references, Sequence
    )
    loc_property = _check_loc_properties(collection, loc_properties=loc_property)[0]
    loc_property_var = loc_property + "_var"

    ddof = 0 if biased else 1
    results_df = pd.DataFrame()
    results_df["localization_count"] = collection.data.loc[:, "localization_count"]
    results_df[loc_property_var] = [
        reference_.data[loc_property].var(ddof=ddof)
        for reference_ in collection.references
    ]

    return results_df


# The specific analysis classes


@dataclass(repr=False)
class PositionVarianceExpectationResults:
    values: pd.DataFrame = field(default_factory=pd.DataFrame)
    # with index being reference_index
    # with columns: loc_property, other_loc_property, value_to_expectation_ratio
    grouped: pd.DataFrame = field(default_factory=pd.DataFrame)
    # with index being other_loc_property
    # with columns: loc_property_mean, loc_property_std, expectation


class PositionVarianceExpectation(_Analysis):
    """
    Analyze variation of localization properties
    in relation to expected variations.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    loc_property: str
        The localization property to analyze.
    expectation : int | float | Mapping[str, Any] | pd.Series[Any] | None
        The expected variance for all or each localization property.
        The expected variance equals the squared localization precision
        for localization position coordinates.
    biased : bool
        Flag to use biased or
        unbiased (Bessel-corrected) variance

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : PositionVarianceExpectationResults
        Computed results.
    distribution_statistics : Distribution_stats, None
        Distribution parameters derived from MLE fitting of results.
    """

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        loc_property: str = "position_x",
        expectation: int | float | Mapping[str, Any] | pd.Series[Any] | None = None,
        biased: bool = True,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.results = None
        self.distribution_statistics = None

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

        loc_property = self.parameter["loc_property"] + "_var"
        other_loc_property = "localization_count"

        self.results = PositionVarianceExpectationResults()

        self.results.values = _property_variances(
            collection=locdata,
            loc_property=self.parameter["loc_property"],
            biased=self.parameter["biased"],
        )

        grouped = self.results.values.groupby(other_loc_property)

        self.results.grouped[loc_property + "_mean"] = grouped.mean()
        self.results.grouped[loc_property + "_std"] = grouped.std()

        expectation = self.parameter["expectation"]
        if expectation is None:
            self.results.grouped["expectation"] = pd.NA
        elif isinstance(expectation, (int, float)) and self.parameter["biased"] is True:
            n_samples = self.results.grouped.index
            expectation = biased_variance(variance=expectation, n_samples=n_samples)
            expectation = pd.Series(data=expectation, index=n_samples)
            self.results.grouped["expectation"] = expectation
        elif isinstance(expectation, (int, float)):
            self.results.grouped["expectation"] = expectation
        elif isinstance(expectation, pd.Series):
            indices = self.results.grouped.index
            self.results.grouped["expectation"] = expectation.loc[indices]
        elif isinstance(expectation, Mapping):
            self.results.grouped["expectation"] = [
                expectation[index_] for index_ in self.results.grouped.index
            ]

        self.results.values["expectation"] = self.results.grouped.loc[
            self.results.values[other_loc_property], "expectation"
        ].to_numpy()

        self.results.values["value_to_expectation_ratio"] = (
            self.results.values[loc_property] / self.results.values["expectation"]
        )

        return self

    def plot(self, ax: mpl.axes.Axes | None = None, **kwargs: Any) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        variances as function of localization counts.

        Parameters
        ----------
        ax
            The axes on which to show the image
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

        self.results.values.plot(
            kind="scatter",
            alpha=0.2,
            x="localization_count",
            y=self.parameter["loc_property"] + "_var",
            color="gray",
            ax=ax,
            **dict(dict(legend=False), **kwargs),
        )

        self.results.grouped.plot(
            kind="line",
            marker=".",
            x=None,
            y=self.parameter["loc_property"] + "_var_mean",
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
            title="Position Variance Expectation",
            xlabel="localization_count",
            ylabel=self.parameter["loc_property"] + "_var",
        )

        return ax

    def hist(
        self,
        ax: mpl.axes.Axes | None = None,
        bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
        n_bins: int | Sequence[int] | None = None,
        bin_size: float | Sequence[float] | Sequence[Sequence[float]] | None = None,
        bin_edges: Sequence[float] | Sequence[Sequence[float]] | None = None,
        bin_range: tuple[float, float]
        | Sequence[float]
        | Sequence[Sequence[float]]
        | None = None,
        log: bool = True,
        fit: bool = False,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        2-dimensional histogram of variances and localization counts.

        Parameters
        ----------
        ax
            The axes on which to show the image
        bins
            The bin specification as defined in :class:`Bins`
        bin_edges
            Bin edges for all or each dimension
            with shape (dimension, n_bin_edges).
        bin_range
            Minimum and maximum edge for all or each dimensions
            with shape (2,) or (dimension, 2).
        n_bins
            The number of bins for all or each dimension.
            5 yields 5 bins in all dimensions.
            (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
        bin_size
            The size of bins for all or each bin and for all or each dimension
            with shape (dimension,) or (dimension, n_bins).
            5 would describe bin_size of 5 for all bins in all dimensions.
            ((2, 5),) yield bins of size (2, 5) for one dimension.
            (2, 5) yields bins of size 2 for one dimension and 5 for the other
            dimension.
            ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and
            (1, 3) for the other dimension.
            To specify arbitrary sequence of `bin_size` use `bin_edges` instead.
        log
            Flag for plotting on a log scale.
        fit
            Flag indicating if distribution fit is shown.
            The fit will only be computed if `distribution_statistics`
             is None.
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.pcolormesh`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
            return ax

        fig = ax.get_figure()

        other_loc_property = "localization_count"
        loc_property = self.parameter["loc_property"] + "_var"

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
        fig.colorbar(mesh)  # type:ignore[union-attr]

        self.results.grouped.plot(
            kind="line",
            marker=".",
            x=None,
            y=self.parameter["loc_property"] + "_var_mean",
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
            title="Position Variance Expectation",
            xlabel=other_loc_property,
            ylabel=loc_property,
        )

        if log:
            ax.set(xscale="log", yscale="log")

        if fit:
            raise NotImplementedError

        return ax
