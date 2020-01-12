"""

This module provides functions for rendering locdata objects.

"""

import numpy as np
import matplotlib.pyplot as plt
import fast_histogram
from skimage import exposure

try:
    import mpl_scatter_density
    _has_mpl_scatter_density = True
except ImportError:
    _has_mpl_scatter_density = False

from surepy.constants import COLORMAP_CONTINUOUS, RenderEngine


__all__ = ['adjust_contrast', 'histogram', 'render_2d']


# todo: add DataFrame input
def _coordinate_ranges(locdata, range=None):
    """
    Provide coordinate range for locdata that can be fed into a binning algorithm.

    Parameters
    ----------
    locdata : pandas DataFrame or LocData object
        Localization data.
    range : tuple with shape (dimension, 2) or None or 'zero'
        ((min_x, max_x), (min_y, max_y), ...) range for each coordinate;
        for None (min, max) range are determined from data;
        for 'zero' (0, max) range with max determined from data.

    Returns
    -------
    numpy array of float with shape (dimension, 2)
        A range (min, max) for each available coordinate.
    """
    if range is None:
        ranges_ = locdata.bounding_box.hull.T
    elif isinstance(range, str):
        if range == 'zero':
            ranges_ = locdata.bounding_box.hull
            ranges_[0] = np.zeros(len(ranges_))
            ranges_ = ranges_.T
        else:
            raise ValueError(f'The parameter range={range} is not defined.')
    else:
        if np.ndim(range)!=locdata.dimension:
            raise TypeError(f'The tuple {range} must have the same dimension as locdata which is {locdata.dimension}.')
        else:
            ranges_ = np.asarray(range)

    return ranges_


def _bin_number(range, bin_size):
    """
    Compute the number of bins from bin_size and range.

    Parameters
    ----------
    range : tuple of shape (2,)
        min and max value fo bin edges.
    bin_size : float
        bin size

    Returns
    -------
    int or numpy array of int
    """
    range_ = np.asarray(range)
    bin_size_ = np.asarray(bin_size)

    if np.ndim(range_) == 1 and np.ndim(bin_size_) == 0:
        bin_number = np.ceil((range_[1] - range_[0]) / bin_size_)
    elif np.ndim(range_) > 1 and np.ndim(bin_size_) == 0:
        bin_number = np.ceil((range_[:, 1] - range_[:, 0]) / bin_size_)
    elif np.ndim(range_) > 1 and np.ndim(bin_size_) > 0:
        bin_number = np.ceil((range_[:, 1] - range_[:, 0]) / bin_size_)
    elif np.ndim(range_) == 1 and np.ndim(bin_size_) > 0:
        bin_number = np.ceil((range_[1] - range_[0]) / bin_size_)
    else:
        raise TypeError('Incompatible dimensions.')

    bin_number = bin_number.astype(int)
    return bin_number


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
        For 'rescale' = True kwargs are passed to exposure.rescale_intensity().
        For 'rescale' = 'equal' kwargs are passed to exposure.equalize_hist().

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
        p_low, p_high = np.percentile(img, rescale)
        img = exposure.rescale_intensity(img, in_range=(p_low, p_high))
    else:
        raise TypeError('Set rescale to tuple, None or "equal".')

    return img


    # elif isinstance(rescale, tuple):
    #     minmax = (img.min(), img.max())
    #     print('minmax:', minmax)
    #     rescale_abs = tuple(np.multiply(np.divide(rescale, 100), (minmax[1] - minmax[0])) + minmax[0])
    #     print('rescale_abs:', rescale_abs)
    #     img = exposure.rescale_intensity(img, in_range=rescale_abs)
    #     print(img)


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


def histogram(locdata, loc_properties=None, other_property=None, bins=None, bin_size=10, range=None, rescale=None):
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

    Returns
    -------
    tuple
        (img, range, bins, label)
    """
    # todo: adjust for loc_property input
    range_ = _coordinate_ranges(locdata, range=range)

    if bins is not None and bin_size is not None:
        raise ValueError('Only one of bins and bin_size can be different from None.')
    elif bins is not None and bin_size is None:
        bins_ = bins
    elif bins is None and bin_size is not None:
        bins_ = _bin_number(range_, bin_size)
    else:
        raise ValueError('One of bins or bin_size must be different from None.')

    if loc_properties is None:
        data = locdata.coordinates.T
        labels = list(locdata.coordinate_labels)
    elif isinstance(loc_properties, str) and loc_properties in locdata.coordinate_labels:
        data = locdata.data[loc_properties].values.T
        range_ = range_[locdata.coordinate_labels.index(loc_properties)]
        bins_ = bins_[locdata.coordinate_labels.index(loc_properties)]
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
        if np.ndim(data) == 1:
            img = fast_histogram.histogram1d(data, range=range_, bins=bins_)
        elif data.shape[0] == 2:
            img = fast_histogram.histogram2d(*data, range=range_, bins=bins_)
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
            img = _fast_histo_mean(*data, values, range=range_, bins=bins_)
            img = img.T  # to show image in the same format as scatter plot
        elif data.shape[0] == 3:
            raise NotImplementedError
        else:
            raise TypeError('No more than 3 elements in loc_properties are allowed.')
        labels.append(other_property)
    else:
        raise TypeError(f'No valid property name {other_property}.')

    if rescale:
        img = adjust_contrast(img, rescale)

    return img, range_, bins_, labels


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

    img, range_, bins_, label = histogram(locdata, loc_properties, other_property, bins, bin_size, range, rescale)

    mappable = ax.imshow(img, origin='low', extent=[*range_[0], *range_[1]], cmap=cmap, **kwargs)
    ax.set(title='Image ({:.0f} nm per bin)'.format(bin_size),
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

    Notes:
    ------
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
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

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

    a = mpl_scatter_density.ScatterDensityArtist(ax, *data, c=values, origin='low', extent=[*range_[0], *range_[1]],
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


def render_2d(locdata, render_engine=RenderEngine.MPL, **kwargs):
    if render_engine == RenderEngine.MPL:
        return render_2d_mpl(locdata, **kwargs)
    elif render_engine == RenderEngine.MPL_SCATTER_DENSITY:
        return render_2d_scatter_density(locdata, **kwargs)
