"""

This module provides functions for rendering locdata objects.

"""
import warnings
from math import isclose
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import fast_histogram
from matplotlib import pyplot as plt
from skimage import exposure
import scipy.signal.windows

from surepy.data.rois import Roi
from surepy.data.region import RoiRegion
from surepy.constants import LOCDATA_ID, COLORMAP_CONTINUOUS, RenderEngine, RENDER_ENGINE
from surepy.constants import _has_mpl_scatter_density, _has_napari
from surepy.data.rois import _MplSelector
from surepy.data.aggregate import histogram
from surepy.data.properties.locdata_statistics import ranges

if _has_mpl_scatter_density: import mpl_scatter_density
if _has_napari: import napari


__all__ = ['render_2d', 'render_2d_mpl', 'render_2d_scatter_density', 'render_2d_napari', 'scatter_2d_mpl',
           'apply_window']


def render_2d_mpl(locdata, loc_properties=None, other_property=None,
                  bins=None, n_bins=None, bin_size=10, bin_edges=None, bin_range=None,
                  rescale=None,
                  ax=None, cmap=COLORMAP_CONTINUOUS, cbar=True, colorbar_kws=None, **kwargs):
    """
    Render localization data into a 2D image by binning x,y-coordinates into regular bins.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    loc_properties : list or None
        Localization properties to be grouped into bins. If None The coordinate_values of locdata are used.
    other_property : str or None
        Localization property (columns in locdata.data) that is averaged in each pixel. If None localization counts are
        shown.
    bins : int or sequence or `Bins` or `boost_histogram.axis.Axis` or None
        The bin specification as defined in :class:`Bins`
    bin_edges : tuple, list, numpy.ndarray of float with shape (n_dimensions, n_bin_edges) or None
        Array of bin edges for all or each dimension.
    n_bins : int, list, tuple or numpy.ndarray or None
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size : float, list, tuple or numpy.ndarray or None
        The size of bins in units of locdata coordinate units for all or each dimension.
        5 would describe bin_size of 5 for all bins in all dimensions.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other dimension.
        To specify arbitrary sequence of `bin_sizes` use `bin_edges` instead.
    bin_range : tuple or tuple of tuples of float with shape (n_dimensions, 2) or None or 'zero'
        The data bin_range to be taken into consideration for all or each dimension.
        ((min_x, max_x), (min_y, max_y), ...) bin_range for each coordinate;
        for None (min, max) bin_range are determined from data;
        for 'zero' (0, max) bin_range with max determined from data.
    rescale : True, tuple, False or None, 'equal', or 'unity.
        Rescale intensity values to be within percentile of max and min intensities
        (tuple with upper and lower bounds provided in percent).
        For True intensity values are rescaled to the min and max possible values of the given representation.
        For 'equal' intensity values are rescaled by histogram equalization.
        For 'unity' intensity values are rescaled to (0, 1).
        For None or False no rescaling occurs.
    ax : matplotlib axes
        The axes on which to show the image
    cmap : str or Colormap instance
        The colormap used to map normalized data values to RGBA colors.
    cbar : bool
        If true draw a colorbar. The colobar axes is accessible using the cax property.
    colorbar_kws : dict
        Keyword arguments for `matplotlib.pyplot.colorbar`.

    Other Parameters
    ----------------
    kwargs : dict
        Other parameters passed to matplotlib.axes.Axes.imshow().

    Returns
    -------
    matplotlib Axes
        Axes object with the image.
    """
    # todo: plot empty image if ranges are provided.
    if not len(locdata):
        raise ValueError('Locdata does not contain any data points.')

    # Provide matplotlib axes if not provided
    if ax is None:
        ax = plt.gca()

    hist = histogram(locdata, loc_properties, other_property,
                                              bins, n_bins, bin_size, bin_edges, bin_range,
                                              rescale)

    mappable = ax.imshow(hist.data, origin='lower', extent=[*hist.bins.bin_range[0], *hist.bins.bin_range[1]],
                         cmap=cmap, **kwargs)

    ax.set(
        title=hist.labels[-1],
        xlabel=hist.labels[0],
        ylabel=hist.labels[1]
        )

    if cbar:
        if colorbar_kws is None:
            plt.colorbar(mappable, ax=ax)
        else:
            plt.colorbar(mappable, **colorbar_kws)

    return ax


def render_2d_scatter_density(locdata, loc_properties=None, other_property=None, bin_range=None,
                              ax=None, cmap=COLORMAP_CONTINUOUS, cbar=True, colorbar_kws=None, **kwargs):
    """
    Render localization data into a 2D image by binning x,y-coordinates into regular bins.

    Prepare matplotlib axes with image.

    Note
    ----
    To rescale intensity values use norm keyword.

    Parameters
    ----------
    locdata : LocData object
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
    ax : matplotlib axes
        The axes on which to show the image
    cmap : str or Colormap instance
        The colormap used to map normalized data values to RGBA colors.
    cbar : bool
        If true draw a colorbar. The colobar axes is accessible using the cax property.
    colorbar_kws : dict
        Keyword arguments for `matplotlib.pyplot.colorbar`.

    Other Parameters
    ----------------
    kwargs : dict
        Other parameters passed to mpl_scatter_density.ScatterDensityArtist().

    Returns
    -------
    matplotlib Axes
        Axes object with the image.
    """
    if not _has_mpl_scatter_density:
        raise ImportError('mpl-scatter-density is required.')

    # todo: plot empty image if ranges are provided.
    if not len(locdata):
        raise ValueError('Locdata does not contain any data points.')

    # Provide matplotlib axes if not provided
    if ax is None:
        ax = plt.gca()
        fig = ax.get_figure()
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density', label='scatter_density')

    if loc_properties is None:
        data = locdata.coordinates.T
        labels = list(locdata.coordinate_labels)
    elif isinstance(loc_properties, str) and loc_properties in locdata.coordinate_labels:
        data = locdata.data[loc_properties].values.T
        labels = list(loc_properties)
    elif isinstance(loc_properties, (list, tuple)):
        for prop in loc_properties:
            if prop not in locdata.coordinate_labels:
                raise ValueError(f'{prop} is not a valid property in locdata.')
        data = locdata.data[list(loc_properties)].values.T
        labels = list(loc_properties)
    else:
        raise ValueError(f'{loc_properties} is not a valid property in locdata.')

    if bin_range is None or isinstance(bin_range, str):
        bin_range_ = ranges(locdata, loc_properties=labels, special=bin_range)
    else:
        bin_range_ = bin_range

    if other_property is None:
        # histogram data by counting points
        if data.shape[0] == 2:
            values = None
        else:
            raise TypeError('Only 2D data is supported.')
        labels.append('counts')
    elif other_property in locdata.data.columns:
        # histogram data by averaging values
        if data.shape[0] == 2:
            # here color serves as weight since it is averaged over all points before binning.
            values = locdata.data[other_property].values.T
        else:
            raise TypeError('Only 2D data is supported.')
        labels.append(other_property)
    else:
        raise TypeError(f'No valid property name {other_property}.')

    a = mpl_scatter_density.ScatterDensityArtist(ax, *data, c=values, origin='lower', extent=[*bin_range_[0], *bin_range_[1]],
                                                 cmap=cmap, **kwargs)
    mappable = ax.add_artist(a)
    ax.set_xlim(*bin_range_[0])
    ax.set_ylim(*bin_range_[1])

    ax.set(title=labels[-1],
           xlabel=labels[0],
           ylabel=labels[1]
           )

    if cbar:
        if colorbar_kws is None:
            plt.colorbar(mappable, ax=ax, label=labels[-1])
        else:
            plt.colorbar(mappable, **colorbar_kws)

    return ax


def render_2d_napari(locdata, loc_properties=None, other_property=None,
                     bins=None, n_bins=None, bin_size=10, bin_edges=None, bin_range=None,
                     rescale=None, viewer=None, cmap='viridis', **kwargs):
    """
    Render localization data into a 2D image by binning x,y-coordinates into regular bins.
    Render the data using napari.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    loc_properties : list or None
        Localization properties to be grouped into bins. If None The coordinate_values of locdata are used.
    other_property : str or None
        Localization property (columns in locdata.data) that is averaged in each pixel. If None localization counts are
        shown.
    bins : int or sequence or `Bins` or `boost_histogram.axis.Axis` or None
        The bin specification as defined in :class:`Bins`
    bin_edges : tuple, list, numpy.ndarray of float with shape (n_dimensions, n_bin_edges) or None
        Array of bin edges for all or each dimension.
    n_bins : int, list, tuple or numpy.ndarray or None
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size : float, list, tuple or numpy.ndarray or None
        The size of bins in units of locdata coordinate units for all or each dimension.
        5 would describe bin_size of 5 for all bins in all dimensions.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other dimension.
        To specify arbitrary sequence of `bin_sizes` use `bin_edges` instead.
    bin_range : tuple or tuple of tuples of float with shape (n_dimensions, 2) or None or 'zero'
        The data bin_range to be taken into consideration for all or each dimension.
        ((min_x, max_x), (min_y, max_y), ...) bin_range for each coordinate;
        for None (min, max) bin_range are determined from data;
        for 'zero' (0, max) bin_range with max determined from data.
    rescale : True, tuple, False or None, 'equal', or 'unity.
        Rescale intensity values to be within percentile of max and min intensities
        (tuple with upper and lower bounds provided in percent).
        For True intensity values are rescaled to the min and max possible values of the given representation.
        For 'equal' intensity values are rescaled by histogram equalization.
        For 'unity' intensity values are rescaled to (0, 1).
        For None or False no rescaling occurs.
    viewer : napari viewer
        The viewer object on which to add the image
    cmap : str or Colormap instance
        The colormap used to map normalized data values to RGBA colors.

    Other Parameters
    ----------------
    kwargs : dict
        Other parameters passed to napari.Viewer().add_image().

    Returns
    -------
    napari Viewer object, namedtuple('Histogram', "data bins labels"): (np.ndarray, `Bins`, list)
        viewer, histogram
    """
    if not _has_napari:
        raise ImportError('Function requires napari.')

    # todo: plot empty image if ranges are provided.
    if not len(locdata):
        raise ValueError('Locdata does not contain any data points.')

    # Provide napari viewer if not provided
    if viewer is None:
        viewer = napari.Viewer(axis_labels=loc_properties)

    hist = histogram(locdata, loc_properties, other_property,
                                              bins, n_bins, bin_size, bin_edges, bin_range,
                                              rescale)

    viewer.add_image(hist.data, name=f'LocData {LOCDATA_ID}', colormap=cmap, **kwargs)
    return viewer, hist


def render_2d(locdata, render_engine=RENDER_ENGINE, **kwargs):
    """
    Wrapper function to render localization data into a 2D image.
    For complete signatures see render_2d_mpl or corresponding functions.
    """
    if render_engine == RenderEngine.MPL:
        return render_2d_mpl(locdata, **kwargs)
    elif _has_mpl_scatter_density and render_engine == RenderEngine.MPL_SCATTER_DENSITY:
        return render_2d_scatter_density(locdata, **kwargs)
    elif _has_napari and render_engine == RenderEngine.NAPARI:
        return render_2d_napari(locdata, **kwargs)


def scatter_2d_mpl(locdata, ax=None, index=True, text_kwargs={}, **kwargs):
    """
    Scatter plot of locdata elements with text marker for each element.

    Parameters
    ----------
    locdata : LocData object
       Localization data.
    ax : matplotlib axes
       The axes on which to show the plot
    index : bool
       Flag indicating if element indices are shown.
    text_kwargs : dict
       Keyword arguments for `matplotlib.axes.Axes.text()`.

    Other Parameters
    ----------------
    kwargs : dict
       Other parameters passed to matplotlib.axes.Axes.scatter().

    Returns
    -------
    matplotlib Axes
       Axes object with the image.
    """
    if not len(locdata):
        raise ValueError('Locdata does not contain any data points.')

    # Provide matplotlib axes if not provided
    if ax is None:
        ax = plt.gca()

    coordinates = locdata.coordinates
    sc = ax.scatter(*coordinates.T, **dict({'marker': '+', 'color': 'grey'}, **kwargs))

    # plot element number
    if index:
        for centroid, marker in zip(coordinates, locdata.data.index.values):
            ax.text(*centroid, marker, **dict({'color': 'grey', 'size': 20}, **text_kwargs))

    ax.set(
           xlabel='position_x',
           ylabel='position_y'
           )

    return ax


def apply_window(image, window_function='tukey', **kwargs):
    """
    Apply window function to image.

    Parameters
    ----------
    image : ndarray
        Image
    window_function : str
        Window function to apply. One of 'tukey', 'hann' or any other in `scipy.signal.windows`.

    Other parameters
    ----------------
    kwargs : dict
        Other parameters passed to the `scipy.signal.windows` window function.
    """
    window_func = getattr(scipy.signal.windows, window_function)
    windows = [window_func(M, **kwargs) for M in image.shape]

    result = image.astype('float64')
    result *= windows[0]
    result *= windows[1][:, None]

    return result


def select_by_drawing_mpl(locdata, region_type='rectangle', **kwargs):
    """
    Select region of interest from rendered image by drawing rois.

    Parameters
    ----------
    locdata : LocData object
        The localization data from which to select localization data.
    region_type : str
        rectangle, or ellipse specifying the selection widget to use.

    Other Parameters
    ----------------
    kwargs :
        kwargs as specified for render_2d

    Returns
    -------
    list
        A list of Roi objects

    See Also
    --------
    surepy.scripts.sc_draw_roi_mpl : script for drawing rois
    matplotlib.widgets : selector functions
    """

    fig, ax = plt.subplots(nrows=1, ncols=1)
    render_2d_mpl(locdata, ax=ax, **kwargs)
    selector = _MplSelector(ax, type=region_type)
    plt.show()
    roi_list = [Roi(reference=locdata, region_specs=roi['region_specs'],
                    region_type=roi['region_type']) for roi in selector.rois]
    return roi_list


def _napari_shape_to_RoiRegion(vertices, bin_edges, region_type):
    """
    Convert napari shape to surepy RoiRegion

    Parameters
    ----------
    vertices : ndarray of float
        Sequence of point coordinates as returned by napari
    bin_edges : tuple, list, ndarray of float with shape (n_dimension, n_bin_edges)
        Array of bin edges for each dimension. At this point there are only equally-sized bins allowed.
    region_type : str
        String specifying the selector widget that can be either rectangle, ellipse, or polygon.

    Returns
    -------
    RoiRegion
    """
    # at this point there are only equally-sized bins used.
    bin_sizes = [bedges[1] - bedges[0] for bedges in bin_edges]

    # flip since napari returns vertices with first component representing the horizontal axis
    vertices = np.flip(vertices, axis=1)

    vertices = np.array([bedges[0] + vert * bin_size
                         for vert, bedges, bin_size in zip(vertices.T, bin_edges, bin_sizes)]
                        ).T

    if region_type == 'rectangle':
        if len(set(vertices[:, 0].astype(int))) != 2:
            raise NotImplementedError('Rotated rectangles are not implemented.')
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        corner_x, corner_y = mins
        width, height = maxs - mins
        angle = 0
        region_specs = ((corner_x, corner_y), width, height, angle)

    elif region_type == 'ellipse':
        if len(set(vertices[:, 0].astype(int))) != 2:
            raise NotImplementedError('Rotated ellipses are not implemented.')
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        width, height = maxs - mins
        center_x, center_y = mins[0] + width/2, mins[1] + height/2
        angle = 0
        region_specs = ((center_x, center_y), width, height, angle)

    elif region_type == 'polygon':
        region_specs = np.concatenate([vertices, [vertices[0]]], axis=0)

    else:
        raise TypeError(f' Type {region_type} is not defined in surepy.')

    return RoiRegion(region_specs=region_specs, region_type=region_type)


def select_by_drawing_napari(locdata, bin_size=10, **kwargs):
    """
    Select region of interest from rendered image by drawing rois in napari.

    Parameters
    ----------
    locdata : LocData object
        The localization data from which to select localization data.
    bin_size : float or None
        The bin size in units of locdata coordinate units.

    Other Parameters
    ----------------
    kwargs :
        kwargs as specified for `render_2d_napari()`.

    Returns
    -------
    list of Roi objects

    See Also
    --------
    surepy.scripts.rois : script for drawing rois
    """
    # select roi

    with napari.gui_qt():
        viewer, hist = render_2d_napari(locdata, bin_size=bin_size, **kwargs)

    vertices = viewer.layers['Shapes'].data
    types = viewer.layers['Shapes'].shape_type

    regions = []
    for verts, typ in zip(vertices, types):
        regions.append(_napari_shape_to_RoiRegion(verts, hist.bins.bin_edges, typ))

    roi_list = [Roi(reference=locdata, region_specs=region.region_specs,
                    region_type=region.region_type) for region in regions]
    return roi_list
