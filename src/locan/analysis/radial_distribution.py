"""

Radial distance distribution analysis.

Radial distance distribution analysis is closely related to the pairwise
distance analysis.

The pairwise distance distribution p(r) represents the probability
distribution function to find for any localization at :math: `r = 0`
another localization at distance r.

The radial distribution function (also called pair correlation function) g(r)
represents the pairwise distance distribution function p(r) normalized to the
region measure m(r) of the circular ring or shell with inner radius r and outer
radius :math: `r + \\delta_r`.
and relative to the expected number of localizations per unit area for
complete spatial randomness.

.. math::

   g(r) &= p(r) \\ ( \rho * m(r) )

The region measure depends on the coordinates dimension and is

.. math::

   m(r) &= r * \\delta r \\qquad \\text{in 1D}

   m(r) &= 2 * \\pi * r * \\delta r \\qquad \\text{in 2D}

   m(r) &= 4 * \\pi * r^2 * \\delta r \\qquad \\text{in 3D}

For a spatial homogeneous Poisson process (i.e. complete spatial randomness,
CSR) with intensity :math:`\\rho` (expected number of points per unit area)
the radial distribution function results to :math:`g(r) = 1`.

See Also
---------
:class:`PairDistances`

References
----------
.. [1] Sengupta, P., Jovanovic-Talisman, T., Skoko, D. et al.,
       Probing protein heterogeneity in the plasma membrane using PALM and pair
       correlation analysis.
       Nat Methods 8, 969–975 (2011),
       https://doi.org/10.1038/nmeth.1704

.. [2] J. Schnitzbauer, Y. Wang, S. Zhao, M. Bakalar, T. Nuwal, B. Chen,
       B. Huang,
       Correlation analysis framework for localization-based superresolution
       microscopy.
       Proc. Natl. Acad. Sci. U.S.A. 115 (13) 3219-3224 (2018),
       https://doi.org/10.1073/pnas.1711314115

.. [3] Bohrer, C.H., Yang, X., Thakur, S. et al.
       A pairwise distance distribution correction (DDC) algorithm to
       eliminate blinking-caused artifacts in SMLM.
       Nat Methods 18, 669–677 (2021).
       https://doi.org/10.1038/s41592-021-01154-y
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typing_extensions import TypeVar

from locan import PairDistances

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

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis
from locan.analysis.pair_distances import _pair_distances

__all__: list[str] = [
    "RadialDistribution",
    "RadialDistributionBatch",
    "RadialDistributionResults",
    "RadialDistributionBatchResults",
]

logger = logging.getLogger(__name__)


# The algorithms


def _radial_distribution_function(
    pair_distances: npt.ArrayLike,
    dimension: int,
    n_points: int,
    other_points_density: float,
    bins: int | Sequence[int | float] | str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the radial distribution function from pairwise distances
    between point set A and other point set B.

    Parameters
    ----------
    pair_distances
        The pairwise distances
    dimension
        The dimension of points
    n_points
        The number of points in A
    other_points_density
        The spatial density of points in B
    bins
        Bin specification (or radii) as used in :func:`numpy.histogram`

    Returns
    -------
    tuple[npt.NDArray[np.float64]]
    """
    pair_distances = np.asarray(pair_distances)
    values: npt.NDArray[np.float64]
    bin_edges: npt.NDArray[np.float64]

    values, bin_edges = np.histogram(
        pair_distances,
        bins=bins,
        density=False,
    )
    radii = bin_edges[:-1]
    delta_radii = np.diff(bin_edges)

    # differential_region_measure
    if dimension == 1:
        differential_region_measure = delta_radii
    elif dimension == 2:
        differential_region_measure = np.pi * ((radii + delta_radii) ** 2 - radii**2)
    elif dimension == 3:
        differential_region_measure = (
            4 / 3 * np.pi * ((radii + delta_radii) ** 3 - radii**3)
        )
    else:
        raise NotImplementedError(f"Not implemented for dimension {dimension}")

    # normalize
    factor = 2 / (n_points * other_points_density * differential_region_measure)
    values = values * factor

    return radii, delta_radii, values


# The specific analysis classes


@dataclass(repr=False)
class RadialDistributionResults:
    radii: pd.DataFrame = field(default_factory=pd.DataFrame)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)


class RadialDistribution(_Analysis):
    """
    Compute the radial distribution function within data or
    between data and other_data.

    The algorithm relies on sklearn.metrics.pairwise.

    Parameters
    ----------
    meta
        Metadata about the current analysis routine.
    bins
        Bin specification (or radii) as used in :func:`numpy.histogram`
    pair_distances
        Precomputed pair distances if available.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : RadialDistributionResults
        Computed results.

    See Also
    ---------
    :class:`PairDistances`
    """

    count = 0

    def __init__(
        self,
        bins: int | Sequence[int | float] | str,
        pair_distances: npt.ArrayLike | PairDistances | None = None,
        meta: metadata_analysis_pb2.AMetadata | None = None,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.dimension: int | None = None
        self.results: RadialDistributionResults | None = None

    def compute(self, locdata: LocData, other_locdata: LocData | None = None) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata
           Localization data.
        other_locdata
            Other localization data to be taken as pairs.

        Returns
        -------
        Self
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        self.dimension = locdata.dimension
        localization_count = locdata.properties["localization_count"]
        # setting the region_measure of locdata
        try:
            region_measure = locdata.properties["region_measure"]
        except KeyError:
            region_measure = locdata.properties["region_measure_bb"]

        if self.parameter["pair_distances"] is not None:
            try:
                pair_distances = self.parameter["pair_distances"].results.pair_distance
            except AttributeError:
                pair_distances = self.parameter["pair_distances"]

        if other_locdata is None:
            other_localization_count = localization_count - 1
            other_localization_density = other_localization_count / region_measure
            points = locdata.coordinates
            if self.parameter["pair_distances"] is None:
                pair_distances = _pair_distances(points=points)

        else:
            # check dimensions
            if other_locdata.dimension != self.dimension:
                raise TypeError(
                    "Dimensions for locdata and other_locdata must be identical."
                )

            # setting the localization density of locdata
            try:
                other_localization_density = other_locdata.properties[
                    "localization_density"
                ]
            except KeyError:
                other_localization_density = other_locdata.properties[
                    "localization_density_bb"
                ]

            points = locdata.coordinates
            other_points = other_locdata.coordinates
            if self.parameter["pair_distances"] is None:
                pair_distances = _pair_distances(
                    points=points, other_points=other_points
                )

        radii, delta_radii, values = _radial_distribution_function(
            pair_distances=pair_distances,
            dimension=self.dimension,
            n_points=localization_count,
            other_points_density=other_localization_density,
            bins=self.parameter["bins"],
        )

        self.results = RadialDistributionResults()
        self.results.radii = pd.DataFrame(
            data={"delta_radii": delta_radii}, index=pd.Index(radii, name="radius")
        )
        self.results.data = pd.DataFrame(
            data={"rdf": values}, index=pd.Index(radii, name="radius")
        )

        return self

    def hist(
        self,
        ax: mpl.axes.Axes | None = None,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing results.

        Parameters
        ----------
        ax
            The axes on which to show the image.
        kwargs
            Other parameters passed to :func:`matplotlib.bar`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
            return ax

        bin_edges = np.append(
            arr=self.results.data.index,
            values=self.results.data.index[-1]
            + self.results.radii["delta_radii"].iloc[-1],
        )

        ax.hist(
            x=self.results.radii.index,
            bins=bin_edges,  # type: ignore[arg-type]
            weights=self.results.data["rdf"],
            **dict(
                dict(
                    label="rdf",
                ),
                **kwargs,
            ),
        )

        ax.set(
            title="Radial Distribution Function g(r)",
            xlabel="radius (nm)",
            ylabel="g(r)",
        )
        ax.legend(loc="best")

        return ax


T_RadialDistributionBatch = TypeVar(
    "T_RadialDistributionBatch", bound="RadialDistributionBatch"
)


@dataclass(repr=False)
class RadialDistributionBatchResults:
    radii: pd.DataFrame = field(default_factory=pd.DataFrame)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)


class RadialDistributionBatch(_Analysis):
    """
    Generate RadialDistribution results from a batch of data.

    See Also
    --------
    :class:`RadialDistribution`

    Parameters
    ----------
    bins
        Bin specification as used in :func:`numpy.histogram`
    meta
        Metadata about the current analysis routine.

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    batch : list[RadialDistribution]
        The generated batch
    dimension : int
        The dimension of original data
    results : RadialDistributionResults
        Computed results.
    """

    def __init__(
        self,
        bins: int | list[int | float] | None = None,
        meta: metadata_analysis_pb2.AMetadata | None = None,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.results = None
        self.batch = None
        self.dimension: int | None = None

    def compute(
        self,
        locdatas: Iterable[LocData],
        other_locdatas: Iterable[LocData] | None = None,
    ) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdatas
            Localization data.
        other_locdatas
            Localization data.

        Returns
        -------
        RadialDistributionBatch
        """
        if other_locdatas is not None:
            raise NotImplementedError(
                "Compute individual RadialDistribution and use RadialDistributionBatch.from_batch"
            )

        radial_distribution_expectation_batch: list[RadialDistribution] = []

        if bool(locdatas):
            dimensions_ = set([item_.dimension for item_ in locdatas])
            if len(dimensions_) == 1:  # check if all are equal
                self.dimension = dimensions_.pop()
            else:
                raise ValueError("The dimensions of all locdata must be the same.")

        for locdata_ in locdatas:
            rd = RadialDistribution(bins=self.parameter["bins"]).compute(locdata_)
            radial_distribution_expectation_batch.append(rd)

        instance = RadialDistributionBatch.from_batch(
            batch=radial_distribution_expectation_batch
        )
        self.dimension = instance.dimension
        self.results = instance.results
        return self

    @classmethod
    def from_batch(
        cls: type[T_RadialDistributionBatch], batch: Sequence[RadialDistribution]
    ) -> T_RadialDistributionBatch:
        batch = [item_ for item_ in batch if item_.results is not None]
        if not bool(batch):
            raise ValueError("The batch is empty.")

        dimension_ = set(item_.dimension for item_ in batch)
        if len(dimension_) == 1:
            dimension = dimension_.pop()
        else:
            raise ValueError("The dimensions of all locdata must be the same.")

        results = RadialDistributionBatchResults()
        assert batch[0].results is not None  # type narrowing # noqa: S101
        radii_ = batch[0].results.radii.index
        if all(np.array_equal(radii_, item_.results.radii.index) for item_ in batch):  # type: ignore[union-attr]
            results.radii.index = radii_  # type: ignore[union-attr]
            results.data.index = radii_  # type: ignore[union-attr]
        else:
            raise ValueError("The radii must be identical for all batch elements.")

        delta_radii_ = batch[0].results.radii.delta_radii
        if all(
            np.array_equal(delta_radii_, item_.results.radii.delta_radii)  # type: ignore[union-attr]
            for item_ in batch
        ):
            results.radii["delta_radii"] = delta_radii_
        else:
            raise ValueError("The bin_widths must be identical for all batch elements.")

        results.data = pd.concat(
            [item_.results.data for item_ in batch if item_.results is not None], axis=1
        )
        instance = cls()
        instance.dimension = dimension
        instance.results = results
        return instance

    def hist(
        self,
        ax: mpl.axes.Axes | None = None,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide step histogram as :class:`matplotlib.axes.Axes` object showing
        results and the mean curve.

        Parameters
        ----------
        ax
            The axes on which to show the image.
        kwargs
            Other parameters passed to :func:`matplotlib.step`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
            return ax

        ax.step(
            x=self.results.data.index,
            y=self.results.data,
            **dict(
                dict(
                    color="lightgrey",
                    alpha=0.2,
                ),
                **kwargs,
            ),
        )

        ax.step(
            x=self.results.data.index,
            y=self.results.data.mean(axis=1),
            **dict(
                dict(
                    color="black",
                    alpha=1,
                    label="mean",
                ),
                **kwargs,
            ),
        )

        ax.step(
            self.results.data.index,
            self.results.data.quantile(0.05, axis=1),
            self.results.data.index,
            self.results.data.quantile(0.95, axis=1),
            **dict(
                dict(
                    color="gray",
                    alpha=1,
                    label="CI",
                    linestyle="dashed",
                ),
                **kwargs,
            ),
        )

        ax.set(
            title="Radial Distribution Function g(r)",
            xlabel="radius (nm)",
            ylabel="g(r)",
        )
        ax.legend(loc="best")

        return ax
