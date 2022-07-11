"""

This module provides functions for rendering locdata objects in 2D.

"""
import logging

import matplotlib.colors as mcolors
import numpy as np
import scipy.signal.windows
from matplotlib import pyplot as plt

from locan import locdata_id
from locan.configuration import COLORMAP_CONTINUOUS, RENDER_ENGINE
from locan.constants import RenderEngine
from locan.data import LocData
from locan.data.aggregate import Bins, _check_loc_properties, histogram
from locan.data.properties.locdata_statistics import ranges
from locan.data.rois import Roi, _MplSelector
from locan.dependencies import HAS_DEPENDENCY
from locan.render.transform import adjust_contrast
from locan.render.utilities import _napari_shape_to_region

if HAS_DEPENDENCY["mpl_scatter_density"]:
    import mpl_scatter_density
if HAS_DEPENDENCY["napari"]:
    import napari


__all__ = [
    "render_2d",
    "render_2d_mpl",
    "render_2d_scatter_density",
    "render_2d_napari",
    "scatter_2d_mpl",
    "apply_window",
    "select_by_drawing_napari",
    "render_2d_rgb_mpl",
    "render_2d_rgb_napari",
]

logger = logging.getLogger(__name__)


def render_2d_mpl(
    locdata,
    loc_properties=None,
    other_property=None,
    bins=None,
    n_bins=None,
    bin_size=10,
    bin_edges=None,
    bin_range=None,
    rescale=None,
    ax=None,
    cmap=COLORMAP_CONTINUOUS,
    cbar=True,
    colorbar_kws=None,
    interpolation="nearest",
    **kwargs,
):
    """
    Render localization data into a 2D image by binning x,y-coordinates into regular bins.

    Parameters
    ----------
    locdata : LocData
        Localization data.
    loc_properties : list, None
        Localization properties to be grouped into bins. If None The coordinate_values of locdata are used.
    other_property : str, None
        Localization property (columns in locdata.data) that is averaged in each pixel. If None localization counts are
        shown.
    bins : int, sequence, Bins, boost_histogram.axis.Axis, None
        The bin specification as defined in :class:`Bins`
    bin_edges : tuple, list, numpy.ndarray of float with shape (dimension, n_bin_edges), None
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
    bin_range : tuple, tuple of tuples of float with shape (dimension, 2), None, 'zero'
        The data bin_range to be taken into consideration for all or each dimension.
        ((min_x, max_x), (min_y, max_y), ...) bin_range for each coordinate;
        for None (min, max) bin_range are determined from data;
        for 'zero' (0, max) bin_range with max determined from data.
    rescale : int, str, locan.Trafo, callable, bool, None
        Transformation as defined in Trafo or by transformation function.
        For None or False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min intensities
        For 'equal' intensity values are rescaled by histogram equalization.
    ax : matplotlib.axes.Axes
        The axes on which to show the image
    cmap : str or Colormap instance
        The colormap used to map normalized data values to RGBA colors.
    cbar : bool
        If true draw a colorbar. The colobar axes is accessible using the cax property.
    colorbar_kws : dict
        Keyword arguments for :func:`matplotlib.pyplot.colorbar`.
    interpolation : str
        Keyword argument for :func:`matplotlib.axes.Axes.imshow`.
    kwargs : dict
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
        locdata,
        loc_properties,
        other_property,
        bins,
        n_bins,
        bin_size,
        bin_edges,
        bin_range,
    )
    data = adjust_contrast(data, rescale)

    mappable = ax.imshow(
        data.T,
        **dict(
            {
                "origin": "lower",
                "extent": [*bins.bin_range[0], *bins.bin_range[1]],
                "cmap": cmap,
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
    locdata,
    loc_properties=None,
    other_property=None,
    bin_range=None,
    ax=None,
    cmap=COLORMAP_CONTINUOUS,
    cbar=True,
    colorbar_kws=None,
    **kwargs,
):
    """
    Render localization data into a 2D image by binning x,y-coordinates into regular bins.

    Prepare :class:`matplotlib.axes.Axes` with image.

    Note
    ----
    To rescale intensity values use norm keyword.

    Parameters
    ----------
    locdata : LocData
        Localization data.
    loc_properties : list or None
        Localization properties to be grouped into bins. If None The coordinate_values of locdata are used.
    other_property : str or None
        Localization property (columns in locdata.data) that is averaged in each pixel. If None localization counts are
        shown.
    bin_range : tuple with shape (dimension, 2) or None or 'zero'
        ((min_x, max_x), (min_y, max_y), ...) bin_range for each coordinate;
        for None (min, max) bin_range are determined from data;
        for 'zero' (0, max) bin_range with max determined from data.
    ax : matplotlib.axes.Axes
        The axes on which to show the image
    cmap : str or Colormap instance
        The colormap used to map normalized data values to RGBA colors.
    cbar : bool
        If true draw a colorbar. The colobar axes is accessible using the cax property.
    colorbar_kws : dict
        Keyword arguments for :func:`matplotlib.pyplot.colorbar`.
    kwargs : dict
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
        ax = fig.add_subplot(
            1, 1, 1, projection="scatter_density", label="scatter_density"
        )

    if loc_properties is None:
        data = locdata.coordinates.T
        labels = list(locdata.coordinate_labels)
    elif (
        isinstance(loc_properties, str) and loc_properties in locdata.coordinate_labels
    ):
        data = locdata.data[loc_properties].values.T
        labels = list(loc_properties)
    elif isinstance(loc_properties, (list, tuple)):
        for prop in loc_properties:
            if prop not in locdata.coordinate_labels:
                raise ValueError(f"{prop} is not a valid property in locdata.")
        data = locdata.data[list(loc_properties)].values.T
        labels = list(loc_properties)
    else:
        raise ValueError(f"{loc_properties} is not a valid property in locdata.")

    if bin_range is None or isinstance(bin_range, str):
        bin_range_ = ranges(locdata, loc_properties=labels, special=bin_range)
    else:
        bin_range_ = bin_range

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
            values = locdata.data[other_property].values.T
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
        cmap=cmap,
        **kwargs,
    )
    mappable = ax.add_artist(a)
    ax.set_xlim(*bin_range_[0])
    ax.set_ylim(*bin_range_[1])

    ax.set(title=labels[-1], xlabel=labels[0], ylabel=labels[1])

    if cbar:
        if colorbar_kws is None:
            plt.colorbar(mappable, ax=ax, label=labels[-1])
        else:
            plt.colorbar(mappable, **colorbar_kws)

    return ax


def render_2d_napari(
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
    Render localization data into a 2D image by binning x,y-coordinates into regular bins.
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
    rescale : int, str, locan.Trafo, callable, bool, None
        Transformation as defined in :class:`locan.Trafo` or by transformation function.
        For None or False no rescaling occurs.
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
    napari.Viewer, Bins
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

    data, bins, labels = histogram(
        locdata,
        loc_properties,
        other_property,
        bins,
        n_bins,
        bin_size,
        bin_edges,
        bin_range,
    )
    data = adjust_contrast(data, rescale)

    viewer.add_image(
        data, **dict({"name": f"LocData {locdata_id}", "colormap": cmap}, **kwargs)
    )
    return viewer, bins


def render_2d(locdata, render_engine=RENDER_ENGINE, **kwargs):
    """
    Wrapper function to render localization data into a 2D image.
    For complete signatures see render_2d_mpl or corresponding functions.
    """
    if render_engine == RenderEngine.MPL:
        return render_2d_mpl(locdata, **kwargs)
    elif (
        HAS_DEPENDENCY["mpl_scatter_density"]
        and render_engine == RenderEngine.MPL_SCATTER_DENSITY
    ):
        return render_2d_scatter_density(locdata, **kwargs)
    elif HAS_DEPENDENCY["napari"] and render_engine == RenderEngine.NAPARI:
        return render_2d_napari(locdata, **kwargs)
    else:
        raise NotImplementedError(f"render_2d is not implemented for {render_engine}.")


def scatter_2d_mpl(locdata, ax=None, index=True, text_kwargs=None, **kwargs):
    """
    Scatter plot of locdata elements with text marker for each element.

    Parameters
    ----------
    locdata : LocData
       Localization data.
    ax : matplotlib.axes.Axes
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

    ax.set(xlabel="position_x", ylabel="position_y")

    return ax


def apply_window(image, window_function="tukey", **kwargs):
    """
    Apply window function to image.

    Parameters
    ----------
    image : numpy.ndarray
        Image
    window_function : str
        Window function to apply. One of 'tukey', 'hann' or any other in `scipy.signal.windows`.
    kwargs : dict
        Other parameters passed to the `scipy.signal.windows` window function.
    """
    window_func = getattr(scipy.signal.windows, window_function)
    windows = [window_func(M, **kwargs) for M in image.shape]

    result = image.astype("float64")
    result *= windows[0]
    result *= windows[1][:, None]

    return result


def select_by_drawing_mpl(locdata, region_type="rectangle", **kwargs):
    """
    Select region of interest from rendered image by drawing rois.

    Parameters
    ----------
    locdata : LocData
        The localization data from which to select localization data.
    region_type : str
        rectangle, or ellipse specifying the selection widget to use.
    kwargs :
        Other parameters as specified for :func:`render_2d`.

    Returns
    -------
    list[Roi]

    See Also
    --------
    :func:`locan.scripts.sc_draw_roi_mpl` : script for drawing rois
    matplotlib.widgets : selector functions
    """

    fig, ax = plt.subplots(nrows=1, ncols=1)
    render_2d_mpl(locdata, ax=ax, **kwargs)
    selector = _MplSelector(ax, type=region_type)
    plt.show()
    roi_list = [
        Roi(reference=locdata, region_specs=roi["region_specs"], region=roi["region"])
        for roi in selector.rois
    ]
    return roi_list


def select_by_drawing_napari(locdata, napari_run=True, **kwargs):
    """
    Select region of interest from rendered image by drawing rois in napari.

    Rois will be created from shapes in napari.viewer.layers['Shapes'].

    Parameters
    ----------
    locdata : LocData
        The localization data from which to select localization data.
    napari_run : bool
        If `True` napari.run is called (set to `False` for testing).
    kwargs : dict
        Other parameters passed to :func:`render_2d_napari`.

    Returns
    -------
    list[Roi]

    See Also
    --------
    :func:`locan.scripts.rois` : script for drawing rois
    """
    # select roi
    viewer, bins = render_2d_napari(locdata, **kwargs)
    if napari_run:
        napari.run()

    vertices = viewer.layers["Shapes"].data
    types = viewer.layers["Shapes"].shape_type

    regions = []
    for verts, typ in zip(vertices, types):
        regions.append(_napari_shape_to_region(verts, bins.bin_edges, typ))

    roi_list = [Roi(reference=locdata, region=reg) for reg in regions]
    return roi_list


def render_2d_rgb_mpl(
    locdatas,
    loc_properties=None,
    other_property=None,
    bins=None,
    n_bins=None,
    bin_size=10,
    bin_edges=None,
    bin_range=None,
    rescale=None,
    ax=None,
    interpolation="nearest",
    **kwargs,
):
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
    locdatas : list[LocData]
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
    rescale : int, str, locan.Trafo, callable, bool, None
        Transformation as defined in :class:`locan.Trafo` or by transformation function.
        For False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min intensities
        For 'equal' intensity values are rescaled by histogram equalization.
    ax : matplotlib.axes.Axes
        The axes on which to show the image
    cmap : str or Colormap instance
        The colormap used to map normalized data values to RGBA colors.
    cbar : bool
        If true draw a colorbar. The colobar axes is accessible using the cax property.
    colorbar_kws : dict
        Keyword arguments for :func:`matplotlib.pyplot.colorbar`.
    interpolation : str
        Keyword argument for :func:`matplotlib.axes.Axes.imshow`.
    kwargs : dict
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


def render_2d_rgb_napari(
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
    Render localization data into a 2D RGB image by binning x,y-coordinates into regular bins.

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
    rescale : int, str, locan.Trafo, callable, bool, None
        Transformation as defined in :class:`locan.Trafo` or by transformation function.
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
    napari.Viewer, Bins
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
    rgb_stack = np.stack([new] * 3, axis=2)

    for i, img in enumerate(imgs):
        rgb_stack[:, :, i] = img

    rgb_stack = np.transpose(rgb_stack, axes=(1, 0, 2))
    viewer.add_image(
        rgb_stack,
        **dict({"name": f"LocData {locdata_id}", "rgb": True}, **kwargs),
    )
    return viewer
