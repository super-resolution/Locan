"""

Analyze geometrical properties of the convex hull of localization coordinates.

Localization coordinates in localization clusters come with a certain variance.
In resolution-limited clusters the variance is determined by the
localization precision. Accordingly, the geometrical properties of convex hulls
also vary and can be analyzed as function of
localization counts to help characterize localization clusters [1]_.

References
----------
.. [1] Ebert V, Eiring P, Helmerich DA, Seifert R, Sauer M, Doose S.
   Convex hull as diagnostic tool in single-molecule localization microscopy.
   Bioinformatics 38(24), 2022, 5421-5429, doi: 10.1093/bioinformatics/btac700.

"""
# todo add fit procedure to estimate variance_estimate
from __future__ import annotations

import importlib.resources as importlib_resources
import logging
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol

import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.special as special

from locan.analysis.analysis_base import _Analysis
from locan.data.aggregate import Bins

if TYPE_CHECKING:
    import matplotlib as mpl

__all__ = ["ConvexHullExpectation"]

logger = logging.getLogger(__name__)


class Collection(Protocol):
    data: pd.DataFrame
    references: Iterable

    def update_convex_hulls_in_references(self):
        pass


class ConvexHullExpectationResource(Enum):
    REGION_MEASURE_2D = "lookup_table_area_2d.npy"
    SUBREGION_MEASURE_2D = "lookup_table_peri_2d.npy"
    REGION_MEASURE_3D = "lookup_table_vol_3d.npy"
    SUBREGION_MEASURE_3D = "lookup_table_area_3d.npy"


class ConvexHullProperty(Enum):
    REGION_MEASURE_2D = auto()
    SUBREGION_MEASURE_2D = auto()
    REGION_MEASURE_3D = auto()
    SUBREGION_MEASURE_3D = auto()
    # might be extended -
    # but the previous entries must correspond to ConvexHullExpectationResource


ConvexHullExpectationValues = namedtuple(
    "ConvexHullExpectationValues", ["n_points", "expectation", "std_pos", "std_neg"]
)


def _get_resource(
    resource_directory: str, resource: str | ConvexHullExpectationResource
) -> ConvexHullExpectationValues:
    """
    Get convex hull property values from resource files produced by
    numerical simulations.

    Notes
    -----
    Original data can be found at
    https://github.com/super-resolution/Ebert-et-al-2022-supplement

    Parameters
    ----------
    resource_directory
        Directory with resources
    resource
        Name of resource file within resource_directory

    Returns
    -------
    ConvexHullExpectationValues
    """
    try:
        resource = importlib_resources.files(resource_directory).joinpath(
            resource.value
        )
    except ImportError:  # required for python < 3.9
        resource = importlib_resources.path(
            package=resource_directory, resource=resource.value
        )

    resource_values = np.load(resource)
    n_points = list(range(3, 201))  # hard coded corresponding to n_points in resources
    if len(n_points) != resource_values.shape[1]:
        raise ValueError("The resource files are not correct.")
    return ConvexHullExpectationValues(
        n_points=n_points,
        expectation=resource_values[0],
        std_pos=resource_values[1],
        std_neg=resource_values[2],
    )


# The algorithms


def compute_convex_hull_region_measure_2d(n_points, sigma=1) -> float:
    """
    Compute the expected convex hull area for `n_points` data points
    that are normal distributed in 2-dimensional space
    with standard deviation `sigma`.

    Parameters
    ----------
    n_points : int
        Number of points of convex hull
    sigma : float
        Standard deviation of normal-distributed point coordinates

    Returns
    -------
    float
        Expectation value for convex hull area
    """

    def pdf(x):
        return 1 / (np.sqrt(2 * np.pi)) * np.exp((-1 / 2) * (x**2))

    def cdf(x):
        return 1 / 2 * (1 + special.erf(x / np.sqrt(2)))

    area = (
        3
        * np.pi
        * special.binom(n_points, 3)
        * integrate.quad(
            lambda x: cdf(x) ** (n_points - 3) * pdf(x) ** 3, -np.inf, np.inf
        )[0]
    )
    return sigma**2 * area


def _get_convex_hull_region_measure_2d_expectation(
    n_points, sigma=1
) -> ConvexHullExpectationValues:
    """
    Return the expected convex hull area together with standard deviations
    for `n_points` data points that are normal distributed
    in 2-dimensional space with standard deviation `sigma`.

    Parameters
    ----------
    n_points : array-like
        Number of points of convex hull
    sigma : float
        Standard deviation of normal-distributed point coordinates

    Returns
    -------
    ConvexHullExpectationValues
    """

    convex_hull_expectation_values = _get_resource(
        resource_directory="locan.analysis.resources.convex_hull_expectation",
        resource=ConvexHullExpectationResource.REGION_MEASURE_2D,
    )
    n_points, _, indices = np.intersect1d(
        n_points, convex_hull_expectation_values.n_points, return_indices=True
    )
    result = ConvexHullExpectationValues(
        n_points=n_points,
        expectation=convex_hull_expectation_values.expectation[indices] * sigma**2,
        std_pos=convex_hull_expectation_values.std_pos[indices] * sigma**2,
        std_neg=convex_hull_expectation_values.std_neg[indices] * sigma**2,
    )
    return result


def _get_convex_hull_property_expectation(
    convex_hull_property, n_points, sigma=1
) -> ConvexHullExpectationValues:
    """
    Get the expected convex hull property for `n_points` data points
    that are normal distributed in 2- or 3-dimensional space
    with standard deviation `sigma`.

    Parameters
    ----------
    convex_hull_property : str | ConvexHullProperty
        Choose property and dimension of convex hull
    n_points : array-like
        Number of points of convex hull
    sigma : float
        Standard deviation of normal-distributed point coordinates

    Returns
    -------
    ConvexHullExpectationValues
        Expectation values for convex hull property
    """
    convex_hull_expectation_values = _get_resource(
        resource_directory="locan.analysis.resources.convex_hull_expectation",
        resource=ConvexHullExpectationResource[convex_hull_property.name],
    )
    n_points, _, indices = np.intersect1d(
        n_points, convex_hull_expectation_values.n_points, return_indices=True
    )
    result = ConvexHullExpectationValues(
        n_points=n_points,
        expectation=convex_hull_expectation_values.expectation[indices] * sigma**2,
        std_pos=convex_hull_expectation_values.std_pos[indices] * sigma**2,
        std_neg=convex_hull_expectation_values.std_neg[indices] * sigma**2,
    )
    return result


# The specific analysis classes


@dataclass(repr=False)
class ConvexHullExpectationResults:
    values: pd.DataFrame = field(default_factory=pd.DataFrame)
    # with index being reference_index
    # with columns: loc_property, other_loc_property, value_to_expectation_ratio
    grouped: pd.DataFrame = field(default_factory=pd.DataFrame)
    # with index being other_loc_property
    # with columns: loc_property_mean, loc_property_std, expectation


class ConvexHullExpectation(_Analysis):
    """
    Analyze geometrical properties of the convex hull of localization
    coordinates in relation to expected values.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    convex_hull_property : str
        One of 'region_measure_ch' (i.e. area or volume)
        or 'subregion_measure_ch' (i.e. circumference or surface.)
    expected_variance : float | Iterable[float] | None
        The expected variance for all or each localization property.
        The expected variance equals the squared localization precision
        for localization position coordinates.

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : ConvexHullExpectationResults
        Computed results.
    distribution_statistics : Distribution_stats, None
        Distribution parameters derived from MLE fitting of results.
    """

    def __init__(
        self,
        meta=None,
        convex_hull_property="region_measure_ch",
        expected_variance=None,
    ):
        if convex_hull_property not in ["region_measure_ch", "subregion_measure_ch"]:
            raise TypeError(
                "convex_hull_property must be one of "
                "[region_measure_ch, subregion_measure_ch]"
            )
        super().__init__(
            meta=meta,
            convex_hull_property=convex_hull_property,
            expected_variance=expected_variance,
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

        loc_property = self.parameter["convex_hull_property"]

        self.results = ConvexHullExpectationResults()
        try:
            self.results.values = locdata.data.loc[
                :, ["localization_count", loc_property]
            ]
        except KeyError:
            locdata.update_convex_hulls_in_references()
            self.results.values = locdata.data.loc[
                :, ["localization_count", loc_property]
            ]

        grouped = self.results.values.groupby("localization_count")

        self.results.grouped[loc_property + "_mean"] = grouped.mean()
        self.results.grouped[loc_property + "_std"] = grouped.std()

        convex_hull_property_ = loc_property[:-2] + f"{locdata.dimension}d"
        convex_hull_property_ = ConvexHullProperty[convex_hull_property_.upper()]

        if self.expected_variance is None:
            self.results.grouped["expectation"] = pd.NA
            self.results.grouped["expectation_std_pos"] = pd.NA
            self.results.grouped["expectation_std_neg"] = pd.NA
        else:
            convex_hull_expectation_values = _get_convex_hull_property_expectation(
                n_points=self.results.grouped.index,
                convex_hull_property=convex_hull_property_,
                sigma=np.sqrt(self.expected_variance),
            )
            self.results.grouped[
                "expectation"
            ] = convex_hull_expectation_values.expectation
            self.results.grouped[
                "expectation_std_pos"
            ] = convex_hull_expectation_values.std_pos
            self.results.grouped[
                "expectation_std_neg"
            ] = convex_hull_expectation_values.std_neg

        self.results.values["expectation"] = self.results.grouped.loc[
            self.results.values["localization_count"], "expectation"
        ].to_numpy()

        self.results.values["value_to_expectation_ratio"] = (
            self.results.values[loc_property] / self.results.values["expectation"]
        )

        return self

    def plot(self, ax=None, **kwargs) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        convex_hull_property as function of localization counts.

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
            x="localization_count",
            y=self.parameter["convex_hull_property"],
            color="gray",
            ax=ax,
            **dict(dict(legend=False), **kwargs),
        )

        self.results.grouped.plot(
            kind="line",
            marker=".",
            x=None,
            y=self.parameter["convex_hull_property"] + "_mean",
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

        if not self.results.grouped.expectation_std_pos.isna().all():
            df = (
                self.results.grouped.expectation
                + self.results.grouped.expectation_std_pos
            )
            df.plot(
                kind="line",
                linestyle=":",
                color="gray",
                ax=ax,
            )

        if not self.results.grouped.expectation_std_neg.isna().all():
            df = (
                self.results.grouped.expectation
                - self.results.grouped.expectation_std_neg
            )
            df.plot(
                kind="line",
                linestyle=":",
                color="gray",
                ax=ax,
            )

        ax.set(
            title="Convex Hull Expectation",
            xlabel="localization_count",
            ylabel=self.parameter["convex_hull_property"],
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
        2-dimensional histogram of convex_hull_property
        and localization counts.

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

        other_loc_property = "localization_count"
        loc_property = self.parameter["convex_hull_property"]

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
            y=self.parameter["convex_hull_property"] + "_mean",
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

        if not self.results.grouped.expectation_std_pos.isna().all():
            df = (
                self.results.grouped.expectation
                + self.results.grouped.expectation_std_pos
            )
            df.plot(
                kind="line",
                linestyle=":",
                color="gray",
                ax=ax,
            )

        if not self.results.grouped.expectation_std_neg.isna().all():
            df = (
                self.results.grouped.expectation
                - self.results.grouped.expectation_std_neg
            )
            df.plot(
                kind="line",
                linestyle=":",
                color="gray",
                ax=ax,
            )

        ax.set(
            title="Convex Hull Expectation",
            xlabel=other_loc_property,
            ylabel=loc_property,
        )
        if log:
            ax.set(xscale="log", yscale="log")

        if fit:
            raise NotImplementedError

        return ax
