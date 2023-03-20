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
# todo compute variance_relative_to_expectation_standardized
# todo: add std of variance estimates to plot and hist
# todo add fit procedure to estimate variance_estimate
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from locan.analysis.analysis_base import _Analysis
from locan.data.aggregate import Bins, _check_loc_properties

if TYPE_CHECKING:
    import matplotlib as mpl

__all__ = ["PositionVarianceExpectation"]

logger = logging.getLogger(__name__)


class Collection(Protocol):
    data: pd.DataFrame
    references: Iterable


# The algorithms


def _position_variances(
    collection: Collection,
    loc_properties: Iterable[str] | None = None,
    biased: bool = False,
) -> dict:
    loc_properties = _check_loc_properties(collection, loc_properties=loc_properties)

    ddof_ = 0 if biased else 1

    results_dict = {"localization_count": collection.data.localization_count}
    for loc_property_ in loc_properties:
        label = loc_property_ + "_var"
        results_dict[label] = [
            reference_.data[loc_property_].var(ddof=ddof_)
            # provides unbiased variance by default
            for reference_ in collection.references
        ]

    return results_dict


def _expected_variance_biased(localization_counts, expected_variance) -> np.ndarray:
    """
    The expected variance is biased for a limited number of localizations
    according to

    .. math::

        E(var_{biased})=expected_variance * (1-\frac{1}{localization_counts}).

    Parameters
    ----------
    localization_counts : int | Iterable[int]
        Number of localizations that contributed to expected variance
    expected_variance : array-like
        The variance computed from some localization property

    Returns
    -------
    numpy.ndarray
    """
    localization_counts = np.asarray(localization_counts)
    return expected_variance * (1 - 1 / localization_counts)


# The specific analysis classes


@dataclass(repr=False)
class PositionVarianceExpectationResults:
    variances: pd.Series | pd.DataFrame | None = None
    # with index being reference_index
    variances_mean: pd.Series | pd.DataFrame | None = None
    # with index being n_localizations
    variances_std: pd.Series | pd.DataFrame | None = None
    # with index being n_localizations
    variance_relative_to_expectation_standardized: pd.Series | pd.DataFrame | None = (
        None
    )
    # with index being reference_index


class PositionVarianceExpectation(_Analysis):
    """
    Analyze variation of localization properties
    in relation to expected variations.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    loc_properties: Iterable[str] | None
        The localization properties to analyze.
        If None the coordinate_labels are used.
    expected_variance : float | Iterable[float] | None
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
        self, meta=None, loc_properties=None, expected_variance=None, biased=False
    ):
        super().__init__(
            meta=meta,
            loc_properties=loc_properties,
            expected_variance=expected_variance,
            biased=biased,
        )
        self.expected_variance = expected_variance
        self.results = None
        self.distribution_statistics = None

    def compute(self, locdata=None):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
            Localization data.

        Returns
        -------
        Analysis class
            Returns the Analysis class object (self).
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self
        self.results = PositionVarianceExpectationResults()

        self.results.variances = pd.DataFrame.from_dict(
            data=_position_variances(
                collection=locdata,
                loc_properties=self.parameter["loc_properties"],
                biased=self.parameter["biased"],
            )
        )

        grouped = self.results.variances.groupby("localization_count")
        self.results.variances_mean = grouped.mean()
        self.results.variances_std = grouped.std()

        # todo        self.results.variance_relative_to_expectation_standardized = \
        #            self.results.variances / self.expected_variance / self.expected_variance_std

        return self

    def plot(self, ax=None, loc_property=None, **kwargs) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        variances as function of localization counts.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which to show the image
        loc_property : str, list(str)
            The property for which to plot localization precision;
            if None all plots are shown.
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

        # prepare plot
        if loc_property is not None and not isinstance(loc_property, Iterable):
            self.results.variances.plot(
                kind="scatter",
                alpha=0.2,
                x="localization_count",
                y=loc_property,
                ax=ax,
                **dict(dict(legend=False), **kwargs),
            )

        self.results.variances_mean.plot(
            kind="line", marker="o", x=None, y=loc_property, ax=ax, legend=False
        )

        if self.expected_variance is not None:
            x = self.results.variances_mean.index
            if self.parameter["biased"]:
                y = _expected_variance_biased(
                    localization_counts=x, expected_variance=self.expected_variance
                )
            else:
                y = [self.expected_variance] * len(x)
            ax.plot(x, y, color="black", **kwargs)

        ax.set(
            title="Position Variance Expectation",
            xlabel="localization_count",
            ylabel="variance",
        )

        return ax

    def hist(
        self,
        ax=None,
        loc_property=None,
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
        2-dimensional histogram of variances and localization counts.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which to show the image
        loc_property : str | None
            The property for which to plot localization precision;
            if None the first available property is taken.
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
        fit: bool
            Flag indicating if distribution fit is shown.
            The fit will only be computed if `distribution_statistics`
             is None.
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

        if loc_property is None:
            labels_ = [
                "localization_count",
                self.results.variances.columns.drop("localization_count")[0],
            ]
        else:
            labels_ = ["localization_count", loc_property]

        if all(
            item_ is None for item_ in [bins, n_bins, bin_size, bin_edges, bin_range]
        ):
            max_localization_count_ = self.results.variances_mean.index.max()
            max_loc_property_ = self.results.variances[labels_[1]].max()

            if log:
                axes = bh.axis.AxesTuple(
                    (
                        bh.axis.Regular(
                            max_localization_count_,
                            1,
                            max_localization_count_,
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
                            max_localization_count_, 1, max_localization_count_
                        ),
                        bh.axis.Regular(50, 1, max_loc_property_),
                    )
                )
            bins = Bins(bins=axes)
        else:
            try:
                bins = Bins(
                    bins, n_bins, bin_size, bin_edges, bin_range, labels=labels_
                )
            except ValueError as exc:
                # the error is raised again only to adapt the message.
                raise ValueError(
                    "Bin dimension and len of `loc_properties` is incompatible."
                ) from exc
        axes = bins.boost_histogram_axes
        histogram = bh.Histogram(*axes)
        histogram.reset()
        histogram.fill(
            self.results.variances[labels_[0]], self.results.variances[labels_[1]]
        )
        mesh = ax.pcolormesh(*histogram.axes.edges.T, histogram.view().T, **kwargs)
        fig.colorbar(mesh)

        if self.expected_variance is not None:
            x = bins.bin_centers[0]
            if self.parameter["biased"]:
                y = _expected_variance_biased(
                    localization_counts=x, expected_variance=self.expected_variance
                )
            else:
                y = [self.expected_variance] * len(x)
            ax.plot(x, y, color="White", **kwargs)

        ax.set(
            title="Position Variance Expectation", xlabel=labels_[0], ylabel=labels_[1]
        )
        if log:
            ax.set(xscale="log", yscale="log")

        if fit:
            raise NotImplementedError

        return ax
