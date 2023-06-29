"""

This module provides functions for rendering `LocData` objects in 3D.

"""
from __future__ import annotations

import logging

from matplotlib import pyplot as plt

from locan.data import LocData  # noqa: F401 # for typing

__all__: list[str] = ["scatter_3d_mpl"]

logger = logging.getLogger(__name__)


def scatter_3d_mpl(locdata, ax=None, index=True, text_kwargs=None, **kwargs):
    """
    Scatter plot of locdata elements with text marker for each element.

    Parameters
    ----------
    locdata : LocData
       Localization data.
    ax : matplotlib.axes.Axes3D
       The axes on which to show the plot
    index : bool
       Flag indicating if element indices are shown.
    text_kwargs : dict
       Keyword arguments for :func:`matplotlib.axes.Axes.text`.
    kwargs : dict
       Other parameters passed to :func:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    matplotlib.axes.Axes
       Axes object with the image.
    """
    if text_kwargs is None:
        text_kwargs = {}

    # Provide matplotlib.axes.Axes if not provided
    if ax is None:
        ax = plt.gca()

    # return ax if no or single point in locdata
    if len(locdata) < 2:
        if len(locdata) == 1:
            logger.warning("Locdata carries a single localization.")
        return ax

    coordinates = locdata.coordinates
    ax.scatter(*coordinates.T, **dict({"marker": "+", "color": "grey"}, **kwargs))

    # plot element number
    if index:
        for centroid, marker in zip(coordinates, locdata.data.index.values):
            ax.text(
                *centroid, marker, **dict({"color": "grey", "size": 20}, **text_kwargs)
            )

    ax.set(xlabel="position_x", ylabel="position_y", zlabel="position_z")

    return ax
