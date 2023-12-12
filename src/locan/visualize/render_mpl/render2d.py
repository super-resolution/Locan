"""

This module provides functions for rendering locdata objects in 2D.

"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import scipy.signal.windows
from matplotlib import pyplot as plt

from locan.configuration import COLORMAP_DEFAULTS
from locan.data import LocData
from locan.data.aggregate import Bins, histogram
from locan.data.locdata_utils import _check_loc_properties
from locan.data.properties.locdata_statistics import ranges
from locan.dependencies import HAS_DEPENDENCY
from locan.rois.roi import Roi, _MplSelector
from locan.visualize.colormap import ColormapType, get_colormap
from locan.visualize.transform import adjust_contrast

if HAS_DEPENDENCY["mpl_scatter_density"]:
    import mpl_scatter_density

if TYPE_CHECKING:
    import boost_histogram as bh
    import matplotlib as mpl

    from locan.visualize.transform import Trafo


__all__: list[str] = [
    "render_2d_mpl",
    "render_2d_scatter_density",
    "scatter_2d_mpl",
    "apply_window",
    "render_2d_rgb_mpl",
]

logger = logging.getLogger(__name__)


def render_2d_mpl(
    locdata: LocData,
    loc_properties: list[str] | None = None,
    other_property: str | None = None,
    bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
    n_bins: int | Sequence[int] | None = None,
    bin_size: float | Sequence[float] | Sequence[Sequence[float]] | None = 10,
    bin_edges: Sequence[float] | Sequence[Sequence[float]] | None = None,
    bin_range: tuple[float, float]
    | Sequence[float]
    | Sequence[Sequence[float]]
    | Literal["zero", "link"]
    | None = None,
    rescale: int | str | Trafo | Callable[..., Any] | bool | None = None,
    ax: mpl.axes.Axes | None = None,
    cmap: ColormapType = COLORMAP_DEFAULTS["CONTINUOUS"],
    cbar: bool = True,
    colorbar_kws: dict[str, Any] | None = None,
    interpolation: str = "nearest",
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Render localization data into a 2D image by binning x,y-coordinates into
    regular bins.

    Parameters
    ----------
    locdata
        Localization data.
    loc_properties
        Localization properties to be grouped into bins.
        If None The coordinate_values of `locdata` are used.
    other_property
        Localization property (columns in `locdata.data`) that is averaged
        in each pixel.
        If None, localization counts are shown.
    bins
        The bin specification as defined in :class:`Bins`
    bin_edges
        Bin edges for all or each dimension
        with shape (dimension, n_bin_edges).
    bin_range
        Minimum and maximum edge for all or each dimensions
        with shape (2,) or (dimension, 2).
        If None (min, max) ranges are determined from data and returned;
        if 'zero' (0, max) ranges with max determined from data are returned.
        if 'link' (min_all, max_all) ranges with min and max determined from
        all combined data are returned.
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
    rescale
        Transformation as defined in :class:`locan.Trafo` or by
        transformation function.
        For None or False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min
        intensities.
        For 'equal' intensity values are rescaled by histogram equalization.
    ax
        The axes on which to show the image
    cmap
        The Colormap object used to map normalized data values to RGBA colors.
    cbar
        If true draw a colorbar.
        The colobar axes is accessible using the cax property.
    colorbar_kws
        Keyword arguments for :func:`matplotlib.pyplot.colorbar`.
    interpolation
        Keyword argument for :func:`matplotlib.axes.Axes.imshow`.
    kwargs
        Other parameters passed to :func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object with the image.
    """
    # Provide matplotlib.axes.Axes if not provided
    if ax is None:
        ax = plt.gca()

    # return ax if no or single point in locdata
    if len(locdata) < 2:
        if len(locdata) == 1:
            logger.warning("Locdata carries a single localization.")
        return ax

    data, bins, labels = histogram(
        locdata=locdata,
        loc_properties=loc_properties,
        other_property=other_property,
        bins=bins,
        n_bins=n_bins,
        bin_size=bin_size,
        bin_edges=bin_edges,
        bin_range=bin_range,
    )
    data = adjust_contrast(data, rescale)

    mappable = ax.imshow(
        data.T,
        **dict(
            {
                "origin": "lower",
                "extent": [*bins.bin_range[0], *bins.bin_range[1]],
                "cmap": get_colormap(colormap=cmap).matplotlib,
                "interpolation": interpolation,
            },
            **kwargs,
        ),
    )

    ax.set(title=labels[-1], xlabel=labels[0], ylabel=labels[1])

    if cbar:
        if colorbar_kws is None:
            plt.colorbar(mappable, ax=ax)
        else:
            plt.colorbar(mappable, **colorbar_kws)

    return ax


def render_2d_scatter_density(
    locdata: LocData,
    loc_properties: list[str] | None = None,
    other_property: str | None = None,
    bin_range: tuple[float, float]
    | Sequence[float]
    | Sequence[Sequence[float]]
    | Literal["zero", "link"]
    | None = None,
    ax: mpl.axes.Axes | None = None,
    cmap: ColormapType = COLORMAP_DEFAULTS["CONTINUOUS"],
    cbar: bool = True,
    colorbar_kws: dict[str, Any] | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Render localization data into a 2D image by binning x,y-coordinates into regular bins.

    Prepare :class:`matplotlib.axes.Axes` with image.

    Note
    ----
    To rescale intensity values use norm keyword.

    Parameters
    ----------
    locdata
        Localization data.
    loc_properties
        Localization properties to be grouped into bins.
        If None The coordinate_values of `locdata` are used.
    other_property
        Localization property (columns in `locdata.data`) that is averaged
        in each pixel.
        If None, localization counts are shown.
    bin_range
        Minimum and maximum edge for all or each dimensions
        with shape (2,) or (dimension, 2).
        If None (min, max) ranges are determined from data and returned;
        if 'zero' (0, max) ranges with max determined from data are returned.
        if 'link' (min_all, max_all) ranges with min and max determined from
        all combined data are returned.
    ax
        The axes on which to show the image
    cmap
        The Colormap object used to map normalized data values to RGBA colors.
    cbar
        If true draw a colorbar.
        The colobar axes is accessible using the cax property.
    colorbar_kws
        Keyword arguments for :func:`matplotlib.pyplot.colorbar`.
    kwargs
        Other parameters passed to :class:`mpl_scatter_density.ScatterDensityArtist`.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object with the image.
    """
    if not HAS_DEPENDENCY["mpl_scatter_density"]:
        raise ImportError("mpl-scatter-density is required.")

    # Provide matplotlib.axes.Axes if not provided
    if ax is None:
        ax = plt.gca()

    # return ax if no or single point in locdata
    if len(locdata) < 2:
        if len(locdata) == 1:
            logger.warning("Locdata carries a single localization.")
        return ax
    else:
        fig = ax.get_figure()
        ax = fig.add_subplot(  # type: ignore[union-attr]
            1, 1, 1, projection="scatter_density", label="scatter_density"
        )

    if loc_properties is None:
        data = locdata.coordinates.T
        labels = list(locdata.coordinate_keys)
    elif isinstance(loc_properties, str) and loc_properties in locdata.coordinate_keys:
        data = locdata.data[loc_properties].values.T
        labels = list(loc_properties)
    elif isinstance(loc_properties, (list, tuple)):
        for prop in loc_properties:
            if prop not in locdata.coordinate_keys:
                raise ValueError(f"{prop} is not a valid property in locdata.")
        data = locdata.data[list(loc_properties)].values.T
        labels = list(loc_properties)
    else:
        raise ValueError(f"{loc_properties} is not a valid property in locdata.")

    if bin_range is None or isinstance(bin_range, str):
        bin_range_: npt.NDArray[np.float_] = ranges(locdata, loc_properties=labels, special=bin_range)  # type: ignore
    else:
        bin_range_ = bin_range  # type: ignore[assignment]

    if other_property is None:
        # histogram data by counting points
        if data.shape[0] == 2:
            values = None
        else:
            raise TypeError("Only 2D data is supported.")
        labels.append("counts")
    elif other_property in locdata.data.columns:
        # histogram data by averaging values
        if data.shape[0] == 2:
            # here color serves as weight since it is averaged over all points before binning.
            values = locdata.data[other_property].values.T  # type: ignore
        else:
            raise TypeError("Only 2D data is supported.")
        labels.append(other_property)
    else:
        raise TypeError(f"No valid property name {other_property}.")

    a = mpl_scatter_density.ScatterDensityArtist(
        ax,
        *data,
        c=values,
        origin="lower",
        extent=[*bin_range_[0], *bin_range_[1]],
        cmap=get_colormap(colormap=cmap).matplotlib,
        **kwargs,
    )
    mappable = ax.add_artist(a)
    ax.set_xlim(*bin_range_[0])
    ax.set_ylim(*bin_range_[1])

    ax.set(title=labels[-1], xlabel=labels[0], ylabel=labels[1])

    if cbar:
        if colorbar_kws is None:
            plt.colorbar(mappable, ax=ax, label=labels[-1])  # type:ignore[arg-type]
        else:
            plt.colorbar(mappable, **colorbar_kws)  # type:ignore[arg-type]

    return ax


def scatter_2d_mpl(
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
            ax.text(  # type: ignore[call-arg]
                *centroid, marker, **dict({"color": "grey", "size": 20}, **text_kwargs)
            )

    ax.set(xlabel="position_x", ylabel="position_y")

    return ax


def apply_window(
    image: npt.ArrayLike, window_function: str = "tukey", **kwargs: Any
) -> npt.NDArray[np.float_]:
    """
    Apply window function to image.

    Parameters
    ----------
    image
        Image
    window_function
        Window function to apply. One of 'tukey', 'hann' or any other in `scipy.signal.windows`.
    kwargs
        Other parameters passed to the `scipy.signal.windows` window function.
    """
    image = np.asarray(image)
    window_func = getattr(scipy.signal.windows, window_function)
    windows = [window_func(M, **kwargs) for M in image.shape]

    result = image.astype("float64")
    result *= windows[0]
    result *= windows[1][:, None]

    return result


def select_by_drawing_mpl(
    locdata: LocData, region_type: str = "rectangle", **kwargs: Any
) -> list[Roi]:
    """
    Select region of interest from rendered image by drawing rois.

    Parameters
    ----------
    locdata
        The localization data from which to select localization data.
    region_type
        rectangle, or ellipse specifying the selection widget to use.
    kwargs
        Other parameters as specified for :func:`render_2d`.

    Returns
    -------
    list[Roi]

    See Also
    --------
    :func:`locan.scripts.sc_draw_roi_mpl` : script for drawing rois
    matplotlib.widgets : selector functions
    """

    fig, ax = plt.subplots(nrows=1, ncols=1)  # type: ignore
    render_2d_mpl(locdata, ax=ax, **kwargs)
    selector = _MplSelector(ax, type=region_type)
    plt.show()  # type: ignore
    roi_list = [Roi(reference=locdata, region=roi["region"]) for roi in selector.rois]
    return roi_list


def render_2d_rgb_mpl(
    locdatas: list[LocData],
    loc_properties: list[str] | None = None,
    other_property: str | None = None,
    bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
    n_bins: int | Sequence[int] | None = None,
    bin_size: float | Sequence[float] | Sequence[Sequence[float]] | None = 10,
    bin_edges: Sequence[float] | Sequence[Sequence[float]] | None = None,
    bin_range: tuple[float, float]
    | Sequence[float]
    | Sequence[Sequence[float]]
    | Literal["zero", "link"]
    | None = None,
    rescale: int | str | Trafo | Callable[..., Any] | bool | None = None,
    ax: mpl.axes.Axes | None = None,
    interpolation: str = "nearest",
    **kwargs: Any,
) -> mpl.axes.Axes:
    """
    Render localization data into a 2D RGB image by binning x,y-coordinates into regular bins.

    Note
    ----
    For rescale=False no normalization is carried out image intensities are clipped to (0, 1) for float value
    or (0, 255) for integer values according to the matplotlib.imshow behavior.
    For rescale=None we apply a normalization to (min, max) of all intensity values.
    For all other rescale options the normalization is applied to each individual image.

    Parameters
    ----------
    locdatas
        Localization data.
    loc_properties
        Localization properties to be grouped into bins. If None The coordinate_values of `locdata` are used.
    other_property
        Localization property (columns in `locdata.data`) that is averaged in each pixel.
        If None, localization counts are shown.
    bins
        The bin specification as defined in :class:`Bins`
    bin_edges
        Bin edges for all or each dimension
        with shape (dimension, n_bin_edges).
    bin_range
        Minimum and maximum edge for all or each dimensions
        with shape (2,) or (dimension, 2).
        If None (min, max) ranges are determined from data and returned;
        if 'zero' (0, max) ranges with max determined from data are returned.
        if 'link' (min_all, max_all) ranges with min and max determined from
        all combined data are returned.
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
    rescale
        Transformation as defined in :class:`locan.Trafo` or by
        transformation function.
        For None or False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min
        intensities.
        For 'equal' intensity values are rescaled by histogram equalization.
    ax
        The axes on which to show the image
    interpolation
        Keyword argument for :func:`matplotlib.axes.Axes.imshow`.
    kwargs
        Other parameters passed to :func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object with the image.
    """
    # Provide matplotlib.axes.Axes if not provided
    if ax is None:
        ax = plt.gca()

    locdata_temp = LocData.concat(locdatas)

    # return ax if no or single point in locdata
    if len(locdata_temp) < 2:
        if len(locdata_temp) == 1:
            logger.warning("Locdata carries a single localization.")
        return ax

    if bin_edges is None:
        _, bins, labels = histogram(
            locdata=locdata_temp,
            loc_properties=loc_properties,
            other_property=other_property,
            bins=bins,
            n_bins=n_bins,
            bin_size=bin_size,
            bin_edges=bin_edges,
            bin_range=bin_range,
        )
    else:
        labels = _check_loc_properties(locdata_temp, loc_properties)
        bins = Bins(bin_edges=bin_edges, labels=labels)

    imgs = [
        histogram(
            locdata=locdata, loc_properties=loc_properties, other_property=other_property, bin_edges=bins.bin_edges  # type: ignore
        ).data
        for locdata in locdatas
    ]

    if rescale is None:
        norm: int | str | Trafo | Callable[..., Any] = mcolors.Normalize(
            vmin=np.min(imgs), vmax=np.max(imgs)
        )
    else:
        norm = rescale
    imgs = [adjust_contrast(img, rescale=norm) for img in imgs]

    new = np.zeros_like(imgs[0])
    rgb_stack = np.stack([new] * 3, axis=2)

    for i, img in enumerate(imgs):
        rgb_stack[:, :, i] = img

    rgb_stack = np.transpose(rgb_stack, axes=(1, 0, 2))
    ax.imshow(
        rgb_stack,
        **dict(
            {
                "origin": "lower",
                "extent": [*bins.bin_range[0], *bins.bin_range[1]],
                "interpolation": interpolation,
            },
            **kwargs,
        ),
    )

    ax.set(title=labels[-1], xlabel=labels[0], ylabel=labels[1])

    return ax
