"""

This module provides functions for rendering `LocData` objects in 3D.

"""
import logging

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from locan import locdata_id
from locan.data import LocData
from locan.constants import RenderEngine
from locan.configuration import COLORMAP_CONTINUOUS, RENDER_ENGINE
from locan.dependencies import HAS_DEPENDENCY
from locan.data.aggregate import histogram, Bins, _check_loc_properties
from locan.render.transform import adjust_contrast

if HAS_DEPENDENCY["napari"]:
    import napari


__all__ = ["render_3d", "render_3d_napari", "scatter_3d_mpl", "render_3d_rgb_napari"]

logger = logging.getLogger(__name__)


def render_3d_napari(
    locdata,
    loc_properties=None,
    other_property=None,
    bins=None,
    n_bins=None,
    bin_size=10,
    bin_edges=None,
    bin_range=None,
    rescale=None,
    viewer=None,
    cmap="viridis",
    **kwargs,
):
    """
    Render localization data into a 3D image by binning x,y,z-coordinates into regular bins.
    Render the data using napari.

    Parameters
    ----------
    locdata : LocData
        Localization data.
    loc_properties : list, None
        Localization properties to be grouped into bins. If None The coordinate_values of `locdata` are used.
    other_property : str, None
        Localization property (columns in `locdata.data`) that is averaged in each pixel.
        If None, localization counts are shown.
    bins : int, sequence, Bins, boost_histogram.axis.Axis, None
        The bin specification as defined in :class:`Bins`
    bin_edges : tuple, list, numpy.ndarray[float] with shape (dimension, n_bin_edges), None
        Array of bin edges for all or each dimension.
    n_bins : int, list, tuple, numpy.ndarray, None
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size : float, list, tuple, numpy.ndarray, None
        The size of bins in units of `locdata` coordinate units for all or each dimension.
        5 would describe bin_size of 5 for all bins in all dimensions.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other dimension.
        To specify arbitrary sequence of `bin_sizes` use `bin_edges` instead.
    bin_range : tuple, tuple[tuples[float]] with shape (dimension, 2), str, None
        The data bin_range to be taken into consideration for all or each dimension.
        ((min_x, max_x), (min_y, max_y), ...) bin_range for each coordinate;
        for None (min, max) bin_range are determined from data;
        for 'zero' (0, max) bin_range with max determined from data.
    rescale : int, str, Trafo, callable, bool, None
        Transformation as defined in :class:`locan.constants.Trafo` or by transformation function.
        For False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min intensities
        For 'equal' intensity values are rescaled by histogram equalization.
    viewer : napari.Viewer
        The viewer object on which to add the image
    cmap : str, Colormap
        The Colormap object used to map normalized data values to RGBA colors.
    kwargs : dict
        Other parameters passed to :func:`napari.Viewer.add_image`.

    Returns
    -------
    napari.Viewer
    """
    if not HAS_DEPENDENCY["napari"]:
        raise ImportError("Function requires napari.")

    # Provide napari viewer if not provided
    if viewer is None:
        viewer = napari.Viewer()

    # return ax if no or single point in locdata
    if len(locdata) < 2:
        if len(locdata) == 1:
            logger.warning("Locdata carries a single localization.")
        return viewer

    hist = histogram(
        locdata,
        loc_properties,
        other_property,
        bins,
        n_bins,
        bin_size,
        bin_edges,
        bin_range,
    )

    viewer.add_image(hist.data, name=f"LocData {locdata_id}", colormap=cmap, **kwargs)
    return viewer


def render_3d(locdata, render_engine=RENDER_ENGINE, **kwargs):
    """
    Wrapper function to render localization data into a 3D image.
    For complete signatures see render_3d_mpl or corresponding functions.
    """
    if HAS_DEPENDENCY["napari"] and render_engine == RenderEngine.NAPARI:
        return render_3d_napari(locdata, **kwargs)
    else:
        raise NotImplementedError(f"render_3d is not implemented for {render_engine}.")


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


def render_3d_rgb_napari(
    locdatas,
    loc_properties=None,
    other_property=None,
    bins=None,
    n_bins=None,
    bin_size=10,
    bin_edges=None,
    bin_range=None,
    rescale=None,
    viewer=None,
    **kwargs,
):
    """
    Render localization data into a 3D RGB image by binning x,y,z-coordinates into regular bins.

    Note
    ----
    For rescale=False no normalization is carried out image intensities are clipped to (0, 1) for float value
    or (0, 255) for integer values according to the matplotlib.imshow behavior.
    For rescale=None we apply a normalization to (min, max) of all intensity values.
    For all other rescale options the normalization is applied to each individual image.

    Parameters
    ----------
    locdatas : list[LocData]
        Localization data.
    loc_properties : list, None
        Localization properties to be grouped into bins. If None The coordinate_values of `locdata` are used.
    other_property : str, None
        Localization property (columns in `locdata.data`) that is averaged in each pixel.
        If None localization counts are shown.
    bins : int, sequence, Bins, boost_histogram.axis.Axis, None
        The bin specification as defined in :class:`Bins`
    bin_edges : tuple, list, numpy.ndarray[float] with shape (dimension, n_bin_edges), None
        Array of bin edges for all or each dimension.
    n_bins : int, list, tuple, numpy.ndarray, None
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size : float, list, tuple, numpy.ndarray, None
        The size of bins in units of locdata coordinate units for all or each dimension.
        5 would describe bin_size of 5 for all bins in all dimensions.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other dimension.
        To specify arbitrary sequence of `bin_sizes` use `bin_edges` instead.
    bin_range : tuple, tuple of tuples of float with shape (dimension, 2), str, None
        The data bin_range to be taken into consideration for all or each dimension.
        ((min_x, max_x), (min_y, max_y), ...) bin_range for each coordinate;
        for None (min, max) bin_range are determined from data;
        for 'zero' (0, max) bin_range with max determined from data.
    rescale : int, str, Trafo, callable, bool, None
        Transformation as defined in :class:`locan.constants.Trafo` or by transformation function.
        For None or False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min intensities
        For 'equal' intensity values are rescaled by histogram equalization.
    viewer : napari.Viewer
        The viewer object on which to add the image
    kwargs : dict
        Other parameters passed to :func:`napari.Viewer.add_image`.

    Returns
    -------
    napari.Viewer
    """
    if not HAS_DEPENDENCY["napari"]:
        raise ImportError("Function requires napari.")

    # Provide napari viewer if not provided
    if viewer is None:
        viewer = napari.Viewer()

    locdata_temp = LocData.concat(locdatas)

    # return viewer if no or single point in locdata
    if len(locdata_temp) < 2:
        if len(locdata_temp) == 1:
            logger.warning("Locdata carries a single localization.")
        return viewer

    if bin_edges is None:
        _, bins, labels = histogram(
            locdata_temp,
            loc_properties,
            other_property,
            bins,
            n_bins,
            bin_size,
            bin_edges,
            bin_range,
        )
    else:
        labels = _check_loc_properties(locdata_temp, loc_properties)
        bins = Bins(bin_edges=bin_edges, labels=labels)

    imgs = [
        histogram(
            locdata, loc_properties, other_property, bin_edges=bins.bin_edges
        ).data
        for locdata in locdatas
    ]

    if rescale is None:
        norm = mcolors.Normalize(vmin=np.min(imgs), vmax=np.max(imgs))
    else:
        norm = rescale
    imgs = [adjust_contrast(img, rescale=norm) for img in imgs]

    new = np.zeros_like(imgs[0])
    rgb_stack = np.stack([new] * 3, axis=3)
    for i, img in enumerate(imgs):
        rgb_stack[:, :, :, i] = img
    rgb_stack = np.transpose(rgb_stack, axes=(2, 1, 0, 3))

    viewer.add_image(rgb_stack, name=f"LocData {locdata_id}", rgb=True, **kwargs)
    return viewer
