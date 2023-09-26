"""
Localization-density variation analysis to characterize localization cluster.

The existence of clusters is tested by analyzing variations in cluster area and
localization density within clusters.
The analysis routine follows the ideas in [1]_ and [2]_.

Note
----
The analysis procedure is in an exploratory state and has not been fully
developed and tested.

References
----------
.. [1] Baumgart F., Arnold AM., Leskovar K., Staszek K., Fölser M., Weghuber J., Stockinger H., Schütz GJ.,
   Varying label density allows artifact-free analysis of membrane-protein nanoclusters.
   Nat Methods. 2016 Aug;13(8):661-4. doi: 10.1038/nmeth.3897
.. [2] Spahn C., Herrmannsdörfer F., Kuner T., Heilemann M.
   Temporal accumulation analysis provides simplified artifact-free analysis of membrane-protein nanoclusters.
   Nat Methods. 2016 Nov 29;13(12):963-964. doi: 10.1038/nmeth.406

"""
from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from typing import Any, Literal

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis
from locan.data.cluster.clustering import cluster_hdbscan
from locan.data.filter import random_subset
from locan.data.hulls import ConvexHull
from locan.data.locdata import LocData

__all__: list[str] = ["AccumulationClusterCheck"]

logger = logging.getLogger(__name__)

# The algorithms


def _accumulation_cluster_check_for_single_dataset(
    locdata: LocData,
    region_measure: float,
    algorithm: Callable[..., tuple[LocData, LocData]] = cluster_hdbscan,  # type: ignore
    algo_parameter: dict[str, Any] | None = None,
    hull: Literal["bb", "ch"] = "bb",
) -> tuple[float, float, float]:
    """
    Compute localization density, relative area coverage by the clusters
    (eta), average density of localizations
    within apparent clusters (rho) for a single localization dataset.
    """
    # localization density
    localization_density = len(locdata) / region_measure

    # compute clusters
    if algo_parameter is None:
        algo_parameter = {}
    noise, clust = algorithm(locdata, **algo_parameter)

    if len(clust) == 0:
        # return localization_density, eta, rho
        return np.nan, np.nan, np.nan

    # compute cluster regions and densities
    if hull == "bb":
        # Region_measure_bb has been computed upon instantiation
        if "region_measure_bb" not in clust.data.columns:
            # return localization_density, eta, rho
            return np.nan, np.nan, np.nan

        else:
            # relative area coverage by the clusters
            eta = clust.data["region_measure_bb"].sum() / region_measure

            # average_localization_density_in_cluster
            rho = clust.data["localization_density_bb"].mean()

    elif hull == "ch":
        # compute hulls
        Hs = [ConvexHull(ref.coordinates) for ref in clust.references]  # type: ignore[union-attr]
        clust.dataframe = clust.dataframe.assign(
            region_measure_ch=[H.region_measure for H in Hs]
        )

        localization_density_ch = (
            clust.data["localization_count"] / clust.data["region_measure_ch"]
        )
        clust.dataframe = clust.dataframe.assign(
            localization_density_ch=localization_density_ch
        )

        # relative area coverage by the clusters
        eta = clust.data["region_measure_ch"].sum() / region_measure

        # average_localization_density_in_cluster
        rho = clust.data["localization_density_ch"].mean()

    else:
        raise TypeError("Computation for the specified hull is not implemented.")

    return localization_density, eta, rho


def _accumulation_cluster_check(
    locdata: LocData,
    region_measure: Literal["bb", "ch"] = "bb",
    algorithm: Callable[..., tuple[LocData, LocData]] = cluster_hdbscan,  # type: ignore
    algo_parameter: dict[str, Any] | None = None,
    hull: Literal["bb", "ch"] = "bb",
    n_loc: int = 10,
    divide: Literal["random", "sequential"] = "random",
    n_extrapolate: int = 5,
) -> pd.DataFrame:
    """
    Compute localization density, relative area coverage by the clusters
    (eta), average density of localizations within apparent clusters (rho)
    for the sequence of divided localization datasets.
    """
    # total region
    if isinstance(region_measure, str):
        region_measure_ = locdata.properties["region_measure_" + region_measure]
    else:
        region_measure_ = region_measure

    if isinstance(n_loc, (list, tuple, np.ndarray)):
        if max(n_loc) <= len(locdata):
            numbers_loc = n_loc
        else:
            raise ValueError(
                "The bins must be smaller than the total number of localizations in locdata."
            )
    else:
        numbers_loc = np.linspace(0, len(locdata), n_loc + 1, dtype=int)[1:]

    # take random subsets of localizations
    if divide == "random":
        locdatas = [random_subset(locdata, n_points=n_pts) for n_pts in numbers_loc]
        for locd in locdatas:
            locd.reduce(reset_index=True)
    elif divide == "sequential":
        selected_indices = [
            locdata.data.index[range(n_pts)].values for n_pts in numbers_loc
        ]
        locdatas = [
            LocData.from_selection(locdata, indices=idx) for idx in selected_indices
        ]
    else:
        raise TypeError(f"String input {divide} for divide is not valid.")

    results_ = [
        _accumulation_cluster_check_for_single_dataset(
            locd,
            region_measure=region_measure_,
            algorithm=algorithm,
            algo_parameter=algo_parameter,
            hull=hull,
        )
        for locd in locdatas
    ]

    # linear regression to extrapolate rho_0
    results_ = np.asarray(results_)  # type: ignore
    idx = np.all(np.isfinite(results_), axis=1)
    x = results_[idx, 0]  # type: ignore
    # position 0 being localization_density
    y = results_[idx, 2]  # type: ignore
    # position 2 being rho

    fit_coefficients = np.polyfit(x[0:n_extrapolate], y[0:n_extrapolate], deg=1)
    rho_zero = fit_coefficients[-1]

    if rho_zero <= 0.0:
        logger.warning("Extrapolation of rho yields a negative value.")
        rho_zero = 1

    # combine results
    results_ = [np.append(entry, [rho_zero, entry[2] / rho_zero]) for entry in results_]  # type: ignore
    results = pd.DataFrame(
        data=results_,
        columns=["localization_density", "eta", "rho", "rho_0", "rho/rho_0"],
    )
    return results


# The specific analysis classes


class AccumulationClusterCheck(_Analysis):
    """
    Check for the presence of clusters in localization data by analyzing
    variations in cluster area and localization
    density within clusters.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata | None
        Metadata about the current analysis routine.
    region_measure : float | Literal["bb", "ch"]
        Region measure (area or volume) for the support of locdata. String can
         be any of standard hull identifier.
    algorithm : Callable[..., tuple[LocData, LocData]]
        Clustering algorithm.
    algo_parameter : dict
        Dictionary with kwargs for `algorithm`.
    hull : Literal["bb", "ch"]
        Hull computation that is used to compute cluster region measures
        (area or volume).
        The identifier string can be one of the defined hulls.
    n_loc : int | Sequence[int]
        If n_loc is an int, it defines the number of localization subsets into
        which the total number of localizations
        are distributed.
        If n_loc is a sequence, it defines the number of localizations used
        for each localization subset.
    divide: Literal["random", "sequential"]
        Identifier to choose how to partition the localization data.
        For `random` localizations are selected randomly.
        For `sequential` localizations are selected as chuncks of increasing
        size always starting from the first element.
    n_extrapolate : int
        The number of rho values taken to extrapolate rho_zero.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict[str, Any]
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.DataFrame
        Data frame with  localization density, relative area coverage by the
        clusters (eta), average density of localizations within apparent
        clusters (rho), and rho normalized to the extrapolated value of rho for
        localization_density=0 (rho_zero). If the extrapolation of rho yields
        a negative value rho_zero is set to 1.
    """

    count = 0

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        region_measure: float | Literal["bb", "ch"] = "bb",
        algorithm: Callable[..., tuple[LocData, LocData]] = cluster_hdbscan,  # type: ignore
        algo_parameter: dict[str, Any] | None = None,
        hull: Literal["bb", "ch"] = "bb",
        n_loc: int | Sequence[int] = 10,
        divide: Literal["random", "sequential"] = "random",
        n_extrapolate: int = 5,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)

    def compute(self, locdata: LocData) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata
          Localization data that might be clustered.

        Returns
        -------
        Self
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        self.results = _accumulation_cluster_check(locdata, **self.parameter)
        return self

    def plot(self, ax: mpl.axes.Axes | None = None, **kwargs: Any) -> mpl.axes.Axes:
        """
        Provide plot of results as :class:`matplotlib.axes.Axes` object.

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

        self.results.plot(x="eta", y="rho/rho_0", ax=ax, **kwargs)

        ax.set(
            title="",
            xlabel="Relative clustered area",
            ylabel="Normalized localization density within cluster",
        )

        return ax
