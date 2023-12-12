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
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from locan.data.locdata import LocData

import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.integrate as integrate
import scipy.special as special

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis
from locan.data.aggregate import Bins

if TYPE_CHECKING:
    import matplotlib as mpl

__all__: list[str] = ["ConvexHullExpectation", "ConvexHullExpectationBatch"]

logger = logging.getLogger(__name__)


class ConvexHullProperty(Enum):
    REGION_MEASURE_2D = auto()
    SUBREGION_MEASURE_2D = auto()
    REGION_MEASURE_3D = auto()
    SUBREGION_MEASURE_3D = auto()
    # might be extended -
    # but the previous entries must correspond to ConvexHullExpectationResource


ConvexHullExpectationResource = dict(
    REGION_MEASURE_2D="lookup_table_area_2d.npy",
    SUBREGION_MEASURE_2D="lookup_table_peri_2d.npy",
    REGION_MEASURE_3D="lookup_table_vol_3d.npy",
    SUBREGION_MEASURE_3D="lookup_table_area_3d.npy",
)


class ConvexHullExpectationValues(NamedTuple):
    n_points: list[int] | npt.NDArray[np.int_]
    expectation: npt.NDArray[np.int_ | np.float_]
    std_pos: npt.NDArray[np.float_]
    std_neg: npt.NDArray[np.float_]


def _get_resource(
    resource_directory: str, resource: str
) -> ConvexHullExpectationValues:
    """
    Get convex hull property values from resource files produced by
    numerical simulations.

    Note
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
        resource_ = importlib_resources.files(resource_directory).joinpath(resource)
        resource_values = np.load(str(resource_))

    except (AttributeError, TypeError):  # required for python < 3.9
        with importlib_resources.path(
            package=resource_directory, resource=resource
        ) as resource_:
            resource_values = np.load(str(resource_))

    if "2d" in resource:  # hard coded corresponding to n_points in resources
        n_points = list(range(3, 201))
    elif "3d" in resource:
        n_points = list(range(4, 201))
    else:
        raise ValueError("resource is neither '2d' nor '3d'")
    if len(n_points) != resource_values.shape[1]:
        raise ValueError("The resource files are not correct.")
    return ConvexHullExpectationValues(
        n_points=n_points,
        expectation=resource_values[0],
        std_pos=resource_values[1],
        std_neg=resource_values[2],
    )


# The algorithms


def compute_convex_hull_region_measure_2d(n_points: int, sigma: float = 1) -> float:
    """
    Compute the expected convex hull area for `n_points` data points
    that are normal distributed in 2-dimensional space
    with standard deviation `sigma`.

    Parameters
    ----------
    n_points
        Number of points of convex hull
    sigma
        Standard deviation of normal-distributed point coordinates

    Returns
    -------
    float
        Expectation value for convex hull area
    """

    def pdf(x: float) -> float:
        return_value = 1 / (np.sqrt(2 * np.pi)) * np.exp((-1 / 2) * (x**2))
        return float(return_value)

    def cdf(x: float) -> float:
        return_value = 1 / 2 * (1 + special.erf(x / np.sqrt(2)))
        return float(return_value)

    try:
        area = (
            3
            * np.pi
            * special.binom(n_points, 3)
            * integrate.quad(
                lambda x: cdf(x) ** (n_points - 3) * pdf(x) ** 3, -np.inf, np.inf
            )[0]
        )
        return_value = sigma**2 * area
    except ZeroDivisionError:
        return_value = np.nan

    return float(return_value)


def _get_convex_hull_property_expectation(
    convex_hull_property: str | ConvexHullProperty,
    n_points: npt.ArrayLike,
    sigma: float = 1,
) -> ConvexHullExpectationValues:
    """
    Get the expected convex hull property for `n_points` data points
    that are normal distributed in 2- or 3-dimensional space
    with standard deviation `sigma`.

    Parameters
    ----------
    convex_hull_property
        Choose property and dimension of convex hull
    n_points
        Number of points of convex hull
    sigma
        Standard deviation of normal-distributed point coordinates

    Returns
    -------
    ConvexHullExpectationValues
        Expectation values for convex hull property
    """
    try:
        convex_hull_property = convex_hull_property.name.upper()  # type: ignore
    except AttributeError:
        convex_hull_property = convex_hull_property.upper()  # type: ignore
    convex_hull_expectation_values = _get_resource(
        resource_directory="locan.analysis.resources.convex_hull_expectation",
        resource=ConvexHullExpectationResource[convex_hull_property],
    )
    n_points, _, indices = np.intersect1d(
        n_points, convex_hull_expectation_values.n_points, return_indices=True
    )

    if convex_hull_property == "REGION_MEASURE_2D":
        factor = sigma**2
    elif convex_hull_property == "SUBREGION_MEASURE_2D":
        factor = sigma
    elif convex_hull_property == "REGION_MEASURE_3D":
        factor = sigma**3
        logger.warning("The expectation is scaled by sigma^2")
    elif convex_hull_property == "SUBREGION_MEASURE_3D":
        factor = sigma**2
        logger.warning("The expectation is scaled by sigma^2")
    else:
        raise ValueError("convex_hull_property is undefined")

    result = ConvexHullExpectationValues(
        n_points=n_points,
        expectation=convex_hull_expectation_values.expectation[indices] * factor,
        std_pos=convex_hull_expectation_values.std_pos[indices] * factor,
        std_neg=convex_hull_expectation_values.std_neg[indices] * factor,
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
    meta : locan.analysis.metadata_analysis_pb2.AMetadata | None
        Metadata about the current analysis routine.
    convex_hull_property : Literal["region_measure_ch", "subregion_measure_ch"]
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
    parameter : dict[str, Any]
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : ConvexHullExpectationResults | None
        Computed results.
    distribution_statistics : dict[str, Any] | None
        Distribution parameters derived from MLE fitting of results.
    """

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        convex_hull_property: Literal[
            "region_measure_ch", "subregion_measure_ch"
        ] = "region_measure_ch",
        expected_variance: float | Iterable[float] | None = None,
    ) -> None:
        if convex_hull_property not in ["region_measure_ch", "subregion_measure_ch"]:
            raise TypeError(
                "convex_hull_property must be one of "
                "[region_measure_ch, subregion_measure_ch]"
            )
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.expected_variance = expected_variance
        self.results: ConvexHullExpectationResults | None = None
        self.distribution_statistics: dict[str, Any] | None = None
        self._dimension: int | None = None

    def compute(self, locdata: LocData) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
            Localization data.

        Returns
        -------
        Self
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        self.results = ConvexHullExpectationResults()
        assert self.results is not None  # type narrowing # noqa: S101
        loc_property = self.parameter["convex_hull_property"]
        self._dimension = locdata.dimension

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

        convex_hull_property_ = loc_property[:-2] + f"{self._dimension}d"
        convex_hull_property_ = ConvexHullProperty[convex_hull_property_.upper()]

        if self.expected_variance is None:
            self.results.grouped["expectation"] = pd.NA
            self.results.grouped["expectation_std_pos"] = pd.NA
            self.results.grouped["expectation_std_neg"] = pd.NA
        else:
            convex_hull_expectation_values = _get_convex_hull_property_expectation(
                n_points=self.results.grouped.index,
                convex_hull_property=convex_hull_property_,
                sigma=np.sqrt(self.expected_variance),  # type: ignore[arg-type]
            )
            new_dict = dict(
                expectation=convex_hull_expectation_values.expectation,
                expectation_std_pos=convex_hull_expectation_values.std_pos,
                expectation_std_neg=convex_hull_expectation_values.std_neg,
            )
            new_df = pd.DataFrame(
                new_dict, index=convex_hull_expectation_values.n_points
            )
            self.results.grouped = pd.merge(
                self.results.grouped,
                new_df,
                how="outer",
                left_index=True,
                right_index=True,
            )

        self.results.values = pd.merge(
            self.results.values,
            self.results.grouped["expectation"],
            how="outer",
            left_on="localization_count",
            right_index=True,
        )
        self.results.values["value_to_expectation_ratio"] = (
            self.results.values[loc_property] / self.results.values["expectation"]
        )

        return self

    def plot(self, ax: mpl.axes.Axes | None = None, **kwargs: Any) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        convex_hull_property as function of localization counts.

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
        2-dimensional histogram of convex_hull_property
        and localization counts.

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
                            3,
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
                            max_other_loc_property_, 3, max_other_loc_property_
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


class ConvexHullExpectationBatch(_Analysis):
    """
    Analyze geometrical properties of the convex hull of localization
    coordinates in relation to expected values.

    See Also
    --------
    :class:`ConvexHullExpectation`

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    convex_hull_property : Literal['region_measure_ch', 'subregion_measure_ch']
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
        meta: metadata_analysis_pb2.AMetadata | None = None,
        convex_hull_property: Literal[
            "region_measure_ch", "subregion_measure_ch"
        ] = "region_measure_ch",
        expected_variance: float | Iterable[float] | None = None,
    ) -> None:
        if convex_hull_property not in ["region_measure_ch", "subregion_measure_ch"]:
            raise TypeError(
                "convex_hull_property must be one of "
                "[region_measure_ch, subregion_measure_ch]"
            )
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.expected_variance = expected_variance
        self.results = None
        self.batch = None
        self._dimension: int | None = None
        self._class = ConvexHullExpectation(
            convex_hull_property=convex_hull_property,
            expected_variance=expected_variance,
        )
        self.distribution_statistics = None

    def compute(self, locdatas: Iterable[LocData]) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdatas : Iterable[LocData]
            Localization data.

        Returns
        -------
        Self
        """
        convex_hull_expectation_batch: list[ConvexHullExpectation] = []
        dimensions: list[int | None] = []
        for locdata_ in locdatas:
            che = ConvexHullExpectation(
                convex_hull_property=self.parameter["convex_hull_property"],
                expected_variance=self.parameter["expected_variance"],
            ).compute(locdata_)
            convex_hull_expectation_batch.append(che)
            dimensions.append(che._dimension)

        if bool(locdatas):
            dimensions_ = set(dimensions)
            if len(dimensions_) == 1:  # check if all are equal
                self._dimension = dimensions_.pop()
            else:
                raise ValueError("The dimensions of all locdata must be the same.")

        self.from_batch(batch=convex_hull_expectation_batch)
        return self

    def from_batch(
        self, batch: Iterable[ConvexHullExpectation], dimension: int | None = None
    ) -> Self:
        if not bool(batch) or all(item_.results is None for item_ in batch):
            logger.warning("The batch is empty.")
            return self

        loc_property = self.parameter["convex_hull_property"]
        if self._dimension is None:
            if dimension is not None:
                self._dimension = dimension
            else:
                dimension_ = set(item_._dimension for item_ in batch)
                if len(dimension_) == 1:
                    self._dimension = dimension_.pop()

        self.results = ConvexHullExpectationResults()
        self.results.values = pd.concat(
            [item_.results.values for item_ in batch if item_.results is not None],
            ignore_index=True,
        )

        grouped = self.results.values.groupby("localization_count")

        self.results.grouped.loc[:, loc_property + "_mean"] = grouped[
            loc_property
        ].mean()
        self.results.grouped.loc[:, loc_property + "_std"] = grouped[loc_property].std()

        convex_hull_property_ = loc_property[:-2] + f"{self._dimension}d"
        convex_hull_property_ = ConvexHullProperty[convex_hull_property_.upper()]

        if self.expected_variance is None:
            self.results.grouped.loc[:, "expectation"] = pd.NA  # type: ignore
            self.results.grouped.loc[:, "expectation_std_pos"] = pd.NA  # type: ignore
            self.results.grouped.loc[:, "expectation_std_neg"] = pd.NA  # type: ignore
        else:
            convex_hull_expectation_values = _get_convex_hull_property_expectation(
                n_points=self.results.grouped.index,
                convex_hull_property=convex_hull_property_,
                sigma=np.sqrt(self.expected_variance),  # type: ignore
            )
            new_dict = dict(
                expectation=convex_hull_expectation_values.expectation,
                expectation_std_pos=convex_hull_expectation_values.std_pos,
                expectation_std_neg=convex_hull_expectation_values.std_neg,
            )
            new_df = pd.DataFrame(
                new_dict, index=convex_hull_expectation_values.n_points
            )
            self.results.grouped = pd.merge(
                self.results.grouped,
                new_df,
                how="outer",
                left_index=True,
                right_index=True,
            )
        self._class.results = self.results
        return self

    def plot(self, ax: mpl.axes.Axes | None = None, **kwargs: Any) -> mpl.axes.Axes:
        return self._class.plot(ax=ax, **kwargs)

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
        return self._class.hist(
            ax=ax,
            bins=bins,
            n_bins=n_bins,
            bin_size=bin_size,
            bin_edges=bin_edges,
            bin_range=bin_range,
            log=log,
            fit=fit,
            **kwargs,
        )
