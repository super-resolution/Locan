"""

Pairwise distance distribution analysis.

The pairwise distance distribution p(r) - as derived from a histogram of
pairwise distances n_AB(r) - represents the probability distribution function
to find for any localization from A another localization from B at
distance r.

.. math::

   p(r) &= n(r) \\ n_distances

See Also
---------
:class:`RadialDistribution`

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
from sklearn.metrics.pairwise import pairwise_distances

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis

__all__: list[str] = ["PairDistances"]

logger = logging.getLogger(__name__)


# The algorithms


def _pair_distances(
    points: npt.ArrayLike, other_points: npt.ArrayLike | None = None
) -> npt.NDArray[np.float64]:
    # todo make use of pairwise_distances_chunked
    if other_points is None:
        distance_matrix = pairwise_distances(X=points, Y=points)
        distances = np.concat(
            [distance_matrix[i_, i_ + 1 :] for i_ in range(distance_matrix.shape[0])]
        )
    else:
        distance_matrix = pairwise_distances(X=points, Y=other_points)
        distances = distance_matrix.reshape(-1)

    return distances  # type: ignore[no-any-return]


# The specific analysis classes


class PairDistances(_Analysis):
    """
    Compute the pairwise distances within data or the pairwise distances
    between data and other_data.

    The algorithm relies on sklearn.metrics.pairwise.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.DataFrame
        Computed results.
    """

    count = 0

    def __init__(self, meta: metadata_analysis_pb2.AMetadata | None = None) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.dimension: int | None = None
        self.results: pd.DataFrame | None = None

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

        if other_locdata is None:
            points = locdata.coordinates
            distances = _pair_distances(points=points)

        else:
            # check dimensions
            if other_locdata.dimension != self.dimension:
                raise TypeError(
                    "Dimensions for locdata and other_locdata must be identical."
                )

            points = locdata.coordinates
            other_points = other_locdata.coordinates
            distances = _pair_distances(points=points, other_points=other_points)

        self.results = pd.DataFrame({"pair_distance": distances})
        return self

    def hist(
        self,
        ax: mpl.axes.Axes | None = None,
        bins: int | list[int | float] | Literal["auto"] = "auto",
        density: bool = True,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide histogram of all pairwise distances.

        Parameters
        ----------
        ax
            The axes on which to show the image.
        bins
            Bin specification as used in :func:`matplotlib.hist`
        density
            Flag for normalization as used in matplotlib.hist.
            True returns probability density function; None returns
            counts.
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
            self.results["pair_distance"],
            bins=bins,
            density=density,
            **dict(
                dict(
                    label="pair_distance",
                ),
                **kwargs,
            ),
        )

        ax.set(
            title="Pair Distances\n",  # noqa: ISC003
            xlabel="distance (nm)",
            ylabel="pdf" if density else "counts",
        )
        ax.legend(loc="best")

        return ax
