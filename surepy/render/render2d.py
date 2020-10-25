"""

This module provides functions for rendering locdata objects.

"""
import warnings
from math import isclose

import numpy as np
import matplotlib.pyplot as plt
import fast_histogram
from skimage import exposure

from surepy.constants import LOCDATA_ID, COLORMAP_CONTINUOUS, RenderEngine, RENDER_ENGINE
from surepy.constants import _has_mpl_scatter_density, _has_napari
if _has_mpl_scatter_density: import mpl_scatter_density
if _has_napari: import napari
from surepy.render.utilities import _coordinate_ranges, _bin_edges, _bin_edges_to_number, _bin_edges_from_size


__all__ = ['adjust_contrast', 'histogram',
           'render_2d', 'render_2d_mpl', 'render_2d_scatter_density', 'render_2d_napari', 'scatter_2d_mpl']


def adjust_contrast(img, rescale=True, **kwargs):
    """
    Adjust contrast of img by equalization or rescaling all values.

    Parameters
    ----------
    img : array-like
        Values to be adjusted
    rescale : True, tuple, False or None, 'equal', or 'unity.
        Rescale intensity values to be within percentile of max and min intensities
        (tuple with upper and lower bounds provided in percent).
        For True intensity values are rescaled to the min and max possible values of the given representation.
        For 'equal' intensity values are rescaled by histogram equalization.
        For 'unity' intensity values are rescaled to (0, 1).
        For None or False no rescaling occurs.

    Other Parameters
    ----------------
    kwargs : dict
        For 'rescale' = True kwargs are passed to :func:`skimage.exposure.rescale_intensity`.
        For 'rescale' = 'equal' kwargs are passed to :func:`skimage.exposure.equalize_hist`.

    Returns
    -------
    numpy array
    """
    if rescale is None or rescale is False:
        pass
    elif rescale is True:
        img = exposure.rescale_intensity(img, **kwargs)  # scaling to min/max of img intensities
    elif rescale == 'equal':
        img = exposure.equalize_hist(img, **kwargs)
    elif rescale == 'unity':
        img = exposure.rescale_intensity(img *1., **kwargs)
    elif isinstance(rescale, tuple):
        p_low, p_high = np.ptp(img) * np.asarray(rescale) / 100 + img.min()
        img = exposure.rescale_intensity(img, in_range=(p_low, p_high))
    else:
        raise TypeError('Set rescale to tuple, None or "equal".')

    return img


def _fast_histo_mean(x, y, values, bins, range):
    """
    Provide histogram with averaged values for all counts in each bin.

    Parameters
    ----------
    x : array-like
        first coordinate values
    y : array-like
        second coordinate values
    values : int or float
        property to be averaged
    bins : sequence or int or None
        The bin specification as defined in fast_histogram_histogram2d:
            A sequence of arrays describing the monotonically increasing bin edges along each dimension.
            The number of bins for each dimension (nx, ny, … =bins)
    range : tuple with shape (dimension, 2) or None
        range as requested by fast_histogram_histogram2d

    Returns
    -------
    ndarray
    """
    hist_1 = fast_histogram.histogram2d(x, y, range=range, bins=bins)
    hist_w = fast_histogram.histogram2d(x, y, range=range, bins=bins, weights=values)

    with np.errstate(divide='ignore', invalid='ignore'):
        hist_mean = np.true_divide(hist_w, hist_1)
        hist_mean[hist_mean == np.inf] = 0
        hist_mean = np.nan_to_num(hist_mean)

    return hist_mean


def histogram(locdata, loc_properties=None, other_property=None, bins=None, bin_size=10, range=None, rescale=None,
              **kwargs):
    """
    Make histogram of loc_properties by binning all localizations or averaging other_property within each bin.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    loc_properties : list or None
        Localization properties to be grouped into bins. If None The coordinate_values of locdata are used.
    other_property : str or None
        Localization property (columns in locdata.data) that is averaged in each pixel. If None localization counts are
        shown.
    bins : sequence or int or None
        The bin specification as defined in :func:`numpy.histogramdd`:
        A sequence of arrays describing the monotonically increasing bin edges along each dimension.
        The number of bins for each dimension (nx, ny, … =bins)
        The number of bins for all dimensions (nx=ny=…=bins).
    bin_size : float or tuple, list, ndarray with with length equal to that of range (or the number of loc_properties).
        Number of bins to be used in all or each dimension for which a range is provided.
        The bin size in units of locdata coordinate units. Either bins or bin_size must be specified but not both.
    range : tuple with shape (dimension, 2) or None or 'zero'
        ((min_x, max_x), (min_y, max_y), ...) range for each coordinate;
        for None (min, max) range are determined from data;
        for 'zero' (0, max) range with max determined from data.
    rescale : True, tuple, False or None, 'equal', or 'unity.
        Rescale intensity values to be within percentile of max and min intensities
        (tuple with upper and lower bounds provided in percent).
        For True intensity values are rescaled to the min and max possible values of the given representation.
        For 'equal' intensity values are rescaled by histogram equalization.
        For 'unity' intensity values are rescaled to (0, 1).
        For None or False no rescaling occurs.

    Other Parameters
    ----------------
    kwargs : dict
        For 'rescale' = True kwargs are passed to :func:`skimage.exposure.rescale_intensity`.
        For 'rescale' = 'equal' kwargs are passed to :func:`skimage.exposure.equalize_hist`.

    Returns
    -------
    tuple
        (img, range, bin_edges, label)
    """
    # todo: adjust for loc_property input
    range_ = _coordinate_ranges(locdata, range=range)

    if bins is not None and bin_size is not None:
        raise ValueError('Only one of bins and bin_size can be different from None.')
    elif bins is not None and bin_size is None:
        bin_edges = _bin_edges(bins, range_)
    elif bins is None and bin_size is not None:
        bin_edges = _bin_edges_from_size(bin_size, range_)  # the last bin extends the range to have equally-sized bins.
    else:
        raise ValueError('One of bins or bin_size must be different from None.')

    if loc_properties is None:
        data = locdata.coordinates.T
        labels = list(locdata.coordinate_labels)
    elif isinstance(loc_properties, str) and loc_properties in locdata.coordinate_labels:
        data = locdata.data[loc_properties].values.T
        range_ = range_[locdata.coordinate_labels.index(loc_properties)]
        bin_edges = bin_edges[locdata.coordinate_labels.index(loc_properties)]
        labels = list(loc_properties)
    elif isinstance(loc_properties, (list, tuple)):
        for prop in loc_properties:
            if prop not in locdata.coordinate_labels:
                raise ValueError(f'{prop} is not a valid property in locdata.')
        data = locdata.data[list(loc_properties)].values.T
        labels = list(loc_properties)
    else:
        raise ValueError(f'{loc_properties} is not a valid property in locdata.')

    n_bins = _bin_edges_to_number(
        bin_edges)  # at this point only equally sized bins can be forwarded to fast_histogram.

    if other_property is None:
        # histogram data by counting points
        if np.ndim(data) == 1:
            img = fast_histogram.histogram1d(data, range=range_, bins=n_bins)
        elif data.shape[0] == 2:
            img = fast_histogram.histogram2d(*data, range=range_, bins=n_bins)
            img = img.T  # to show image in the same format as scatter plot
        elif data.shape[0] == 3:
            raise NotImplementedError
        else:
            raise TypeError('No more than 3 elements in loc_properties are allowed.')
        labels.append('counts')

    elif other_property in locdata.data.columns:
        # histogram data by averaging values
        if np.ndim(data) == 1:
            raise NotImplementedError
        elif data.shape[0] == 2:
            values = locdata.data[other_property].values
            img = _fast_histo_mean(*data, values, range=range_, bins=n_bins)
            img = img.T  # to show image in the same format as scatter plot
        elif data.shape[0] == 3:
            raise NotImplementedError
        else:
            raise TypeError('No more than 3 elements in loc_properties are allowed.')
        labels.append(other_property)
    else:
        raise TypeError(f'No valid property name {other_property}.')

    if rescale:
        img = adjust_contrast(img, rescale, **kwargs)

    return img, range_, bin_edges, labels


def render_2d_mpl(locdata, loc_properties=None, other_property=None, bins=None, bin_size=10, range=None, rescale=None,
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
    bins : sequence or int or None
        The bin specification as defined in numpy.histogramdd:
            A sequence of arrays describing the monotonically increasing bin edges along each dimension.
            The number of bins for each dimension (nx, ny, … =bins)
            The number of bins for all dimensions (nx=ny=…=bins).
    bin_size : float or None
        The bin size in units of locdata coordinate units. Either bins or bin_size must be specified but not both.
    range : tuple with shape (dimension, 2) or None or 'zero'
        ((min_x, max_x), (min_y, max_y), ...) range for each coordinate;
        for None (min, max) range are determined from data;
        for 'zero' (0, max) range with max determined from data.
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

    img, range_, bin_edges, label = histogram(locdata, loc_properties, other_property, bins, bin_size, range, rescale)

    mappable = ax.imshow(img, origin='lower', extent=[*range_[0], *range_[1]], cmap=cmap, **kwargs)
    if bin_size is not None:
        ax.set(title='Image ({:.0f} nm per bin)'.format(bin_size))

    ax.set(
           xlabel='position_x',
           ylabel='position_y'
           )

    if cbar:
        if colorbar_kws is None:
            plt.colorbar(mappable, ax=ax)
        else:
            plt.colorbar(mappable, **colorbar_kws)

    return ax


def render_2d_scatter_density(locdata, loc_properties=None, other_property=None, range=None,
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
    range : tuple with shape (dimension, 2) or None or 'zero'
        ((min_x, max_x), (min_y, max_y), ...) range for each coordinate;
        for None (min, max) range are determined from data;
        for 'zero' (0, max) range with max determined from data.
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

    # todo: adjust for loc_property input
    range_ = _coordinate_ranges(locdata, range=range)

    if loc_properties is None:
        data = locdata.coordinates.T
        labels = list(locdata.coordinate_labels)
    elif isinstance(loc_properties, str) and loc_properties in locdata.coordinate_labels:
        data = locdata.data[loc_properties].values.T
        range_ = range_[locdata.coordinate_labels.index(loc_properties)]
        labels = list(loc_properties)
    elif isinstance(loc_properties, (list, tuple)):
        for prop in loc_properties:
            if prop not in locdata.coordinate_labels:
                raise ValueError(f'{prop} is not a valid property in locdata.')
        data = locdata.data[list(loc_properties)].values.T
        labels = list(loc_properties)
    else:
        raise ValueError(f'{loc_properties} is not a valid property in locdata.')

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

    a = mpl_scatter_density.ScatterDensityArtist(ax, *data, c=values, origin='lower', extent=[*range_[0], *range_[1]],
                                                 cmap=cmap, **kwargs)
    mappable = ax.add_artist(a)
    ax.set_xlim(*range_[0])
    ax.set_ylim(*range_[1])

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


def render_2d_napari(locdata, loc_properties=None, other_property=None, bins=None, bin_size=10, range=None,
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
    bins : sequence or int or None
        The bin specification as defined in numpy.histogramdd:
            A sequence of arrays describing the monotonically increasing bin edges along each dimension.
            The number of bins for each dimension (nx, ny, … =bins)
            The number of bins for all dimensions (nx=ny=…=bins).
    bin_size : float or None
        The bin size in units of locdata coordinate units. Either bins or bin_size must be specified but not both.
    range : tuple with shape (dimension, 2) or None or 'zero'
        ((min_x, max_x), (min_y, max_y), ...) range for each coordinate;
        for None (min, max) range are determined from data;
        for 'zero' (0, max) range with max determined from data.
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
    napari Viewer object, ndarray, tuple of str
        viewer, bins_edges, label
    """
    if not _has_napari:
        raise ImportError('Function requires napari.')

    # todo: plot empty image if ranges are provided.
    if not len(locdata):
        raise ValueError('Locdata does not contain any data points.')

    # Provide napari viewer if not provided
    if viewer is None:
        viewer = napari.Viewer(axis_labels=loc_properties)

    img, range, bins_edges, label = histogram(locdata, loc_properties, other_property, bins, bin_size, range,
                                              rescale)
    viewer.add_image(img, name=f'LocData {LOCDATA_ID}', colormap=cmap, **kwargs)
    return viewer, bins_edges, label


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
