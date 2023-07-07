"""
Compute subpixel bias in localization data.

Subpixel bias in localization coordinates may arise depending on the
localization algorithm [1]_.

References
----------
.. [1] Gould, T. J., Verkhusha, V. V. & Hess, S. T.,
   Imaging biological structures with fluorescence photoactivation
   localization microscopy. Nat. Protoc. 4 (2009), 291–308.
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, cast

if sys.version_info >= (3, 9):
    from collections.abc import Sequence  # noqa: F401
else:
    from typing import Sequence  # noqa: F401

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from locan.data.locdata import LocData

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from locan.analysis.analysis_base import _Analysis

__all__: list[str] = ["SubpixelBias"]

logger = logging.getLogger(__name__)


# The algorithms


def _subpixel_bias(
    locdata, pixel_size: int | float | Sequence[int | float]
) -> pd.DataFrame:
    coordinate_labels = locdata.coordinate_keys
    coordinates = locdata.coordinates.T

    if np.ndim(pixel_size) == 0:
        pixel_size = cast("int | float", pixel_size)
        pixel_sizes: Sequence[int | float] = [pixel_size] * len(coordinate_labels)
    else:
        pixel_size = cast("Sequence[int | float]", pixel_size)
        if len(pixel_size) != len(coordinate_labels):
            raise TypeError("There must be given a pixel_size for each coordinate.")
        else:
            pixel_sizes = pixel_size

    coordinates_modulo = [
        np.remainder(coordinates_, pixel_size_)
        for coordinates_, pixel_size_ in zip(coordinates, pixel_sizes)
    ]

    results = {
        label_ + "_modulo": values_
        for label_, values_ in zip(coordinate_labels, coordinates_modulo)
    }

    return pd.DataFrame(results)


# The specific analysis classes


class SubpixelBias(_Analysis):
    """
    Check for subpixel bias by computing the modulo of localization coordinates
    for each localization's spatial coordinate in locdata.

    Parameters
    ----------
    pixel_size : int | float | Sequence[int | float]
        Camera pixel size in coordinate units.
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
        The number of localizations per frame or
        the number of localizations per frame normalized to region_measure(hull).
    """

    count = 0

    def __init__(self, meta=None, pixel_size=None) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.results = None

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

        self.results = _subpixel_bias(locdata=locdata, **self.parameter)
        return self

    def hist(self, ax=None, bins="auto", log=True, **kwargs) -> plt.axes.Axes:
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing
        hist(results). Nan entries are ignored.

        Parameters
        ----------
        ax : matplotlib.axes.Axes`
            The axes on which to show the image
        bins : int | Sequence | str
            Bin specifications (passed to :func:`matplotlib.hist`).
        log : Bool
            Flag for plotting on a log scale.
        kwargs : dict
            Other parameters passed to :func:`matplotlib.pyplot.hist`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        ax.hist(
            self.results.dropna(axis=0).to_numpy(),
            bins=bins,
            **dict(dict(density=True, log=log), **kwargs),
            histtype="step",
            label=self.results.columns,
        )
        ax.set(
            title="Subpixel Bias",
            xlabel="position_modulo_pixel_size",
            ylabel="PDF",
        )

        return ax
