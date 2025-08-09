"""

Compute the local density of localizations.

A local density is computed from the number of neighboring localizations
within a specified radius.

"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any, Literal

from locan import Ellipse, Region

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
from sklearn.neighbors import NearestNeighbors

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis

__all__: list[str] = ["LocalDensity"]

logger = logging.getLogger(__name__)


def _overlap_factor(region: Region, other_region: Region) -> float:
    region_measure = region.region_measure
    region_measure_intersection = region.intersection(other_region).region_measure
    return region_measure_intersection / region_measure


def _local_density(
    points: npt.ArrayLike,
    radii: npt.ArrayLike,
    density: bool = True,
    boundary_correction: Region | None = None,
    normalization: float | npt.ArrayLike | None = None,
    other_points: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """
    Compute the number of neighboring localizations for two- or
    three-dimensional data within the given radius.

    Parameters
    ----------
    points
        2D or 3D points on which to estimate local densities.
    radii
        Radii in which to search for neighboring points.
    density : bool
        If true, the number of neighboring points is normalized by the
        local region_measure according to radius;
        else, the absolute number of points is returned.
    boundary_correction
        A region that determines the outer boundary or None for no correction.
    normalization : float | None
        If not None, the number of neighboring points is normalized by this
        number (in addition to the optional density normalization).
    other_points
        2D or 3D points from which to estimate local densities
        (e.g. subset of points). For None other_points
        is set to points (default).

    Returns
    -------
    npt.NDArray[np.float64]
        Local densities with shape (n_radii, n_points,).
    """
    points = np.asarray(points)
    if normalization is not None:
        normalization = np.asarray(normalization)
    if other_points is None:
        other_points = points
    else:
        other_points = np.asarray(other_points)

    dimension = points.shape[-1]
    if density is True and dimension > 3:
        raise ValueError(
            "A region measure for density normalization cannot be be computed for point dimension >3"
        )

    if boundary_correction is not None and boundary_correction.dimension != dimension:
        raise ValueError("Region must have the same dimension as points.")

    radii = np.asarray(radii)

    nn = NearestNeighbors(metric="euclidean").fit(points)

    densities_list = []
    for radius_ in radii:
        indices_list = nn.radius_neighbors(other_points, radius_, return_distance=False)
        densities_ = np.asarray([len(indices_) - 1 for indices_ in indices_list])
        densities_list.append(densities_)
    densities = np.asarray(densities_list)

    if boundary_correction is not None:
        overlap_factors_list = []
        for radius_ in radii:
            if dimension == 2:
                local_regions = (
                    Ellipse(center=point_, width=2 * radius_, height=2 * radius_)
                    for point_ in points
                )
            else:
                raise NotImplementedError
            overlap_factors_ = [
                _overlap_factor(region_, other_region=boundary_correction)
                for region_ in local_regions
            ]
            overlap_factors_list.append(overlap_factors_)
        overlap_factors = np.asarray(overlap_factors_list)

    if density:
        if dimension == 1:
            local_region_measure = radii
        elif dimension == 2:
            local_region_measure = np.pi * radii**2
        elif dimension == 3:
            local_region_measure = 4 / 3 * np.pi * radii**3
        else:
            raise ValueError(
                "A region measure for density normalization cannot be be computed for point dimension >3"
            )
        densities = np.divide(densities.T, local_region_measure).T

    if boundary_correction is not None:
        densities = np.divide(densities, overlap_factors)

    if normalization is not None:
        densities = np.divide(densities, normalization)

    return densities


# The specific analysis classes


class LocalDensity(_Analysis):
    """
    Compute the local density for two- or three-dimensional data.
    A local density is computed from the number of neighboring localizations
    within a specified radius.

    Parameters
    ----------
    radii : npt.ArrayLike
        Radii in which to search for neighboring points.
    density : bool
        If true, the number of neighboring points is normalized by the
        local region_measure according to radius;
        else, the absolute number of points is returned.
    boundary_correction : bool
        If true, the number of points is corrected for boundary effects.
        The boundary is set by the region of locdata.
    normalization : float | npt.ArrayLike | None
        If not None, the number of neighboring points is normalized by this
        number (in addition to the optional density normalization).
    meta : locan.analysis.metadata_analysis_pb2.AMetadata | None
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
        Data frame with radii as index and local densities.
    """

    count = 0

    def __init__(
        self,
        radii: npt.ArrayLike,
        density: bool = True,
        boundary_correction: bool = False,
        normalization: npt.ArrayLike | float | None = None,
        meta: metadata_analysis_pb2.AMetadata | None = None,
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
            Localization data with 2D or 3D coordinates on which to estimate
            local density.
        other_locdata
            Other localization data from which to estimate local density.
            For None other_points is set to points (default).

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
        else:
            other_points = None

        if self.parameter["boundary_correction"]:
            if locdata.region is None:
                raise ValueError(
                    "Locdata must have a valid region for a boundary correction."
                )
            else:
                boundary_correction = locdata.region
        else:
            boundary_correction = None

        densities = _local_density(
            points=points,
            radii=self.parameter["radii"],
            density=self.parameter["density"],
            boundary_correction=boundary_correction,
            normalization=self.parameter["normalization"],
            other_points=other_points,
        )
        self.results = pd.DataFrame(data=densities.T, columns=self.parameter["radii"])
        return self

    def hist(
        self,
        ax: mpl.axes.Axes | None = None,
        bins: int | list[int | float] | Literal["auto"] = "auto",
        density: bool = True,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing hist(results).

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
            Other parameters passed to :func:`matplotlib.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
            return ax

        _values, bin_values, _patches = ax.hist(
            self.results,
            bins=bins,
            density=density,
            **dict(dict(label=self.results.columns, log=False), **kwargs),
        )

        # modify plot
        xlabel = "normalized " if self.parameter["normalization"] is not None else ""
        if self.parameter["density"]:
            xlabel += "density"
        else:
            xlabel += "counts"

        ax.set(
            title="Local Density",
            xlabel=xlabel,
            ylabel="pdf" if density else "counts",
        )
        ax.legend(loc="best")

        return ax
