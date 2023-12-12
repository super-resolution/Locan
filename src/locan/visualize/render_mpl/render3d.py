"""

This module provides functions for rendering `LocData` objects in 3D.

"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from matplotlib import pyplot as plt

if TYPE_CHECKING:
    import matplotlib as mpl

    from locan.data import LocData

__all__: list[str] = ["scatter_3d_mpl"]

logger = logging.getLogger(__name__)


def scatter_3d_mpl(
    locdata: LocData,
    ax: mpl.axes.Axes | None = None,
    index: bool = True,
    text_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Scatter plot of locdata elements with text marker for each element.

    Parameters
    ----------
    locdata
       Localization data.
    ax
       The axes on which to show the plot
    index
       Flag indicating if element indices are shown.
    text_kwargs
       Keyword arguments for :func:`matplotlib.axes.Axes.text`.
    kwargs
       Other parameters passed to :func:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    matplotlib.axes.Axes3D
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
            ax.text(  # type:ignore[call-arg]
                *centroid, marker, **dict({"color": "grey", "size": 20}, **text_kwargs)
            )

    ax.set(xlabel="position_x", ylabel="position_y", zlabel="position_z")

    return ax
