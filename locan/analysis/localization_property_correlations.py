"""
Analyze cross dependencies between localization properties.

Analyze cross dependencies as indicated by the correlation coefficients between any two localization properties.
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import pandas as pd

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    import matplotlib as mpl

    from locan.data.locdata import LocData

import matplotlib.pyplot as plt
import numpy as np

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis
from locan.configuration import COLORMAP_DEFAULTS

__all__: list[str] = ["LocalizationPropertyCorrelations"]

logger = logging.getLogger(__name__)


# The algorithms


def _localization_property_correlations(
    locdata: LocData, loc_properties: list[str] | None = None
) -> pd.DataFrame:
    if loc_properties is None:
        results = locdata.data.corr()
    else:
        results = locdata.data[loc_properties].corr()
    return results


# The specific analysis classes


class LocalizationPropertyCorrelations(_Analysis):
    """
    Compute and analyze correlation coefficients between any two localization
     properties.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    loc_properties : list[str] | None
        Localization properties to be analyzed. If None all are used.

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    parameter : dict[str, Any]
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata | None
        Metadata about the current analysis routine.
    results : pandas.DataFrame | None
        The correlation coefficients..
    """

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        loc_properties: list[str] | None = None,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.results: pd.DataFrame | None = None

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

        self.results = _localization_property_correlations(
            locdata=locdata, **self.parameter
        )
        return self

    def report(self) -> None:
        if self.results is None:
            logger.warning("No results available")
            return

        print("Fit results for:\n")
        print(self.results.model_result.fit_report(min_correl=0.25))
        # print(self.results.fit_results.best_values)

    def plot(
        self,
        ax: mpl.axes.Axes | None = None,
        cbar: bool = True,
        colorbar_kws: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide heatmap of all correlation values as
        :class:`matplotlib.axes.Axes` object.

        Parameters
        ----------
        ax
            The axes on which to show the image
        cbar
            If true draw a colorbar.
        colorbar_kws
            Keyword arguments for :func:`matplotlib.pyplot.colorbar`.
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.imshow`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
            return ax

        im = ax.imshow(
            self.results,
            **dict(
                dict(vmin=-1, vmax=1, cmap=COLORMAP_DEFAULTS["DIVERGING"]), **kwargs
            ),
        )
        columns = self.results.columns

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(columns)))
        ax.set_yticks(np.arange(len(columns)))

        # ensure correct scaling
        ax.set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(columns) + 1) - 0.5, minor=True)
        ax.tick_params(which="minor", bottom=False, left=False)

        # ... and label them with the respective list entries
        ax.set_xticklabels(columns)
        ax.set_yticklabels(columns)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(columns)):
            for j in range(len(columns)):
                ax.text(
                    j,
                    i,
                    round(self.results.values[i, j], 2),
                    ha="center",
                    va="center",
                    color="w",
                )

        ax.set_title("Localization Property Correlations")

        # Create colorbar
        if cbar:
            if colorbar_kws is None:
                cbar_ = ax.figure.colorbar(im)  # type:ignore[union-attr]
            else:
                cbar_ = ax.figure.colorbar(  # type:ignore[union-attr]  # noqa: F841
                    im, **colorbar_kws
                )
            # cbar_.ax.set_ylabel('correlation', rotation=-90, va="bottom")

        ax.figure.tight_layout()  # type:ignore[union-attr]
        return ax
