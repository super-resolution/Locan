"""
Coordinate-based colocalization.

Colocalization is estimated by computing a colocalization index for each
localization using the so-called coordinate-based colocalization
algorithm [1]_.

References
----------
.. [1] Malkusch S, Endesfelder U, Mondry J, Gelléri M, Verveer PJ, Heilemann M.,
   Coordinate-based colocalization analysis of single-molecule localization microscopy data.
   Histochem Cell Biol. 2012, 137(1):1-10.
   doi: 10.1007/s00418-011-0880-5
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

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
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis

__all__: list[str] = ["CoordinateBasedColocalization"]

logger = logging.getLogger(__name__)


# The algorithm


def _coordinate_based_colocalization(
    points: npt.ArrayLike,
    other_points: npt.ArrayLike | None = None,
    radius: int | float = 100,
    n_steps: int = 10,
) -> npt.NDArray[np.float_]:
    """
    Compute a colocalization index for each localization by coordinate-based
    colocalization.

    Parameters
    ----------
    points
        Array of points with shape (n_points, dimensions)
        for which CBC values are computed.

    other_points
        Array of points with shape (n_points, dimensions) to be
        compared with points. If None other_points are set to points.

    radius
        The maximum radius up to which nearest neighbors are determined

    n_steps
        The number of bins from which Spearman correlation is computed.

    Returns
    -------
    npt.NDArray[np.float_]
        An array with coordinate-based colocalization coefficients for each
        input point.
    """
    points = np.asarray(points)
    # sampled radii
    radii = np.linspace(0, radius, n_steps + 1)

    # nearest neighbors within radius
    nneigh_1 = NearestNeighbors(radius=radius, metric="euclidean").fit(points)
    distances_1 = np.array(nneigh_1.radius_neighbors()[0])

    if other_points is None:
        nneigh_2 = NearestNeighbors(radius=radius, metric="euclidean").fit(points)
    else:
        nneigh_2 = NearestNeighbors(radius=radius, metric="euclidean").fit(other_points)
    distances_2 = np.array(nneigh_2.radius_neighbors(points)[0])

    # CBC for each point
    correlation = np.empty(len(points))
    for i, (d_1, d_2) in enumerate(zip(distances_1, distances_2)):
        if len(d_1) and len(d_2):
            # binning
            hist_1 = np.histogram(d_1, bins=radii, range=(0, radius))[0]
            hist_2 = np.histogram(d_2, bins=radii, range=(0, radius))[0]

            # normalization
            values_1 = np.cumsum(hist_1) * radius**2 / radii[1:] ** 2 / len(d_1)
            values_2 = np.cumsum(hist_2) * radius**2 / radii[1:] ** 2 / len(d_2)

            # Spearman rank correlation
            rho, pval = spearmanr(values_1, values_2)
            correlation[i] = rho
        else:
            correlation[i] = np.nan

    # CBC normalization for each point
    max_distances = np.array(
        [np.max(d, initial=0) for d in distances_2]
    )  # max is set to 0 for empty arrays.
    norm_spearmanr = np.exp(-1 * max_distances / radius)
    correlation = correlation * norm_spearmanr

    return correlation


# The specific analysis classes


class CoordinateBasedColocalization(_Analysis):
    """
    Compute a colocalization index for each localization by coordinate-based
    colocalization (CBC).

    The colocalization index is calculated for each localization in `locdata`
    by finding nearest neighbors in `locdata` or `other_locdata` within
    `radius`. A normalized number of nearest neighbors at a certain radius is
    computed for `n_steps` equally-sized steps of increasing radii ranging
    from 0 to `radius`.
    The Spearman rank correlation coefficent is computed for these values and
    weighted by Exp[-nearestNeighborDistance/distanceMax].

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    radius : int | float
        The maximum radius up to which nearest neighbors are determined
    n_steps : int
        The number of bins from which Spearman correlation is computed.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.DataFrame
        Coordinate-based colocalization coefficients for each input point.
    """

    count = 0

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        radius: int | float = 100,
        n_steps: int = 10,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.results = None

    def compute(self, locdata: LocData, other_locdata: LocData | None = None) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata
            Localization data for which CBC values are computed.
        other_locdata
            Localization data to be colocalized.
            If None other_locdata is set to locdata.

        Returns
        -------
        Self
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        points = locdata.coordinates
        if other_locdata is not None:
            other_points = other_locdata.coordinates
            id_ = other_locdata.meta.identifier
        else:
            other_points = None
            id_ = "self"

        self.results = pd.DataFrame(
            {
                f"colocalization_cbc_{id_}": _coordinate_based_colocalization(
                    points, other_points, **self.parameter
                )
            }
        )
        return self

    def hist(
        self,
        ax: mpl.axes.Axes | None = None,
        bins: int | Sequence[int | float] | Literal["auto"] = (-1, -0.3333, 0.3333, 1),
        density: bool = True,
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
            Bin specification as used in :func:`matplotlib.hist`.
        density
            Flag for normalization as used in :func:`matplotlib.hist`.
            True returns probability density function;
            None returns counts.
        kwargs
            Other parameters passed to :func:`matplotlib.hist`.

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
            self.results.iloc[:, 0].values,
            bins=bins,
            density=density,
            label="cbc",
            **kwargs,
        )

        ax.set(
            title="CBC histogram",
            xlabel="colocalization_cbc",
            ylabel="pdf" if density else "counts",
        )
        ax.legend(loc="best")

        return ax
