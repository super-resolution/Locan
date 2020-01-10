"""

This module provides functions for rendering locdata objects.

"""

import numpy as np
import matplotlib.pyplot as plt
import fast_histogram
from skimage import exposure
# import mpl_scatter_density


import surepy.data.properties.locdata_statistics
from surepy.constants import COLORMAP_CONTINUOUS


__all__ = ['render_2d']


def _coordinate_ranges(locdata, ranges='auto'):
    """
    Provide coordinate ranges for locdata that can be fed into a binning algorithm.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    ranges : tuple with shape (dimension, 2) or 'auto' or 'zero'
        ((min_x, max_x), (min_y, max_y), ...) ranges for each coordinate;
        for 'auto' (min, max) ranges are determined from data;
        for 'zero' (0, max) ranges with max determined from data.

    Returns
    -------
    numpy array of float with shape (dimension, 2)
        A range (min, max) for each available coordinate.
    """
    if isinstance(ranges, str):
        if ranges == 'auto':
            ranges_ = locdata.bounding_box.hull.T

        elif ranges == 'zero':
            ranges_ = locdata.bounding_box.hull
            ranges_[0] = np.zeros(len(ranges_))
            ranges_ = ranges_.T
        else:
            raise ValueError(f'The parameter ranges={ranges} is not defined.')
    else:
        if np.ndim(ranges)!=locdata.dimension:
            raise TypeError(f'The tuple ranges must have the same dimension as locdata.')
        else:
            ranges_ = np.asarray(ranges)

    return ranges_





def render_2d(locdata, ax=None, bin_size=10, ranges='auto', rescale=True,
              cmap=COLORMAP_CONTINUOUS, cbar=True, colorbar_kws=None, **kwargs):
    """
    Render localization data into a 2D image by binning x,y-coordinates into regular bins.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    ax : matplotlib axes
        The axes on which to show the image
    bin_size : int or float
        x and y size of bins
    ranges : [[min,max],[min,max]] or 'auto' (default) or 'zero'
        defining the binned region by [min, max] ranges from input;
        for 'auto' by [min, max] ranges from data; for 'zero' by [0, max] ranges from data.
    rescale : True (default), tuple, None, or 'equal'
        rescale intensity values to be within percentile of max and min intensities
        (tuple with upper and lower bounds provided in percent).
        For True intensity values are rescaled to the min and max possible values of the given representation.
        For 'equal' intensity values are rescaled by histogram equalization.
        For None no rescaling occurs.
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

    # Provide matplotlib axes if not provided
    if not len(locdata):
        raise ValueError('Locdata does not contain any data points.')

    if ax is None:
        ax = plt.gca()

    # determine ranges
    ranges_ = _coordinate_ranges(locdata, ranges=ranges)

    # histogram data
    bin_number = np.ceil((ranges_[:, 1] - ranges_[:, 0]) / bin_size)
    bin_number = bin_number.astype(int)
    # bin_centers = np.arange(range[0], range[1], bin_size) + bin_size / 2

    data = locdata.data[['position_x', 'position_y']].values.T
    img = fast_histogram.histogram2d(*data, range=ranges_, bins=bin_number)
    img = img.T  # to show image in the same format as scatter plot

    # contrast adjustment by equalization or rescaling
    if rescale is None or rescale is False:
        pass
    elif rescale is True:
        img = exposure.rescale_intensity(img)  # scaling to min/max of img intensities
    elif isinstance(rescale, tuple):
        minmax = (img.min(), img.max())
        print('minmax:', minmax)
        rescale_abs = tuple(np.multiply(np.divide(rescale, 100), (minmax[1] - minmax[0])) + minmax[0])
        print('rescale_abs:', rescale_abs)
        img = exposure.rescale_intensity(img, in_range=rescale_abs)
        print(img)
    elif rescale == 'equal':
        img = exposure.equalize_hist(img)
    else:
        raise TypeError('Set rescale to tuple, None or "equal".')

    mappable = ax.imshow(img, origin='low', extent=[*ranges_[0], *ranges_[1]], cmap=cmap, **kwargs)
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


###############################################





def _set_bins(range, bin_size):
    ''' Compute bin_number from bin_size. '''
    bin_number = np.ceil((range[:, 1] - range[:, 0]) / bin_size)
    bin_number = bin_number.astype(int)
    return bin_number

def _adjust_contrast(img, rescale):
    ''' Adjust contrast by equalization or rescaling. '''
    if rescale is None:
        pass
    elif isinstance(rescale, str):
        if rescale == 'equal':
            img = exposure.equalize_hist(img)
    else:
        p_low, p_high = np.percentile(img, rescale)
        img = exposure.rescale_intensity(img, in_range=(p_low, p_high))

    return img

def _fast_histo_mean(x, y, values, range, bins):
    hist_1 = fast_histogram.histogram2d(x, y, range=range, bins=bins)
    hist_w = fast_histogram.histogram2d(x, y, range=range, bins=bins, weights=values)

    with np.errstate(divide='ignore', invalid='ignore'):
        hist_mean = np.true_divide(hist_w, hist_1)
        hist_mean[hist_mean == np.inf] = 0
        hist_mean = np.nan_to_num(hist_mean)

    return hist_mean

def _compute_histogram_2D(locdata, property, range, bins):
    ''' Make 2D histogram by simple binning or averaging property within bins.'''
    if property is None:
        # histogram data by couting points
        data = locdata.data[['Position_x', 'Position_y']].values.T
        img = fast_histogram.histogram2d(*data, range=range, bins=bins)
        img = img.T  # to show image in the same format as scatter plot
        label = 'Counts'
    elif property in locdata.data.columns:
        # histogram data by averaging values
        x = locdata.data['Position_x'].values
        y = locdata.data['Position_y'].values
        values = locdata.data[property].values
        img = _fast_histo_mean(x, y, values, range=range, bins=bins)
        img = img.T  # to show image in the same format as scatter plot
        label = property
    else:
        raise TypeError('No valid property name.')
    return img, label


def render2D(locdata, ax=None, property=None, bin_size=10, range='auto', rescale=(2, 98), colorbar=True, show=True,
             cmap='magma', **kwargs):
    """
    Render localization data into a 2D image by binning x,y-coordinates into regular bins.

    Prepare matplotlib axes with image.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    property : str
        Localization property (columns in locdata.data) that is averaged in each pixel. If None localization counts are
        shown.
    bin_size : int or float
        x and y size of bins
    range : [[min,max],[min,max]] or 'auto' or 'zero'
        defining the binned region by [min, max] ranges from input;
        for 'auto' by [min, max] ranges from data; for 'zero' by [0, max] ranges from data.
    rescale : tuple or 'equal'
        rescale intensity values to be within percentile (tuple with upper and lower bounds)
        or equalize histogram ('equal').
    **kwargs :
        imshow and Artist properties.

    Returns
    -------
    matplotlib.image.AxesImage (rtype from imshow)

    Note:
    -----
    We recommend the following colormaps: 'viridis', 'plasma', 'magma', 'inferno', 'hot', 'hsv'.
    """
    range_ = _get_range(locdata, range=range)
    bin_number = _set_bins(range_, bin_size)
    img, label = _compute_histogram_2D(locdata, property=property, range=range_, bins=bin_number)
    img = _adjust_contrast(img, rescale)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    imgax = ax.imshow(img, origin='low', extent=[*range_[0], *range_[1]], cmap=cmap, **kwargs)
    ax.set(title='{} ({:.0f} nm per bin)'.format(property, bin_size),
           xlabel='Position_x',
           ylabel='Position_y'
           )

    if colorbar is True:
        plt.colorbar(imgax, label=label)

    if show is True:
        plt.show()

    return imgax


def render2D_scatter_density(locdata, ax=None, property=None, range='auto', rescale=(2, 98), show=True):
    """
    Render localization data into a 2D image by binning x,y-coordinates into regular bins.

    Prepare matplotlib axes with image.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    property : string
        localization property to be binned. If property is None counts are represented.
    range : [[min,max],[min,max]] or 'auto' or 'zero'
        defining the binned region by [min, max] ranges from input;
        for 'auto' by [min, max] ranges from data; for 'zero' by [0, max] ranges from data.
    rescale : tuple or 'equal'
        rescale intensity values to be within percentile (tuple with upper and lower bounds)
        or equalize histogram ('equal').

    Returns
    -------
    matplotlib.image.AxesImage (rtype from imshow)
        mappable to create colorbar

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

    range_ = _get_range(locdata, range=range)
    data = locdata.data[['Position_x', 'Position_y']].values.T
    # here color serves as weight since it is averaged over all points before binning.
    if property is not None:
        c = locdata.data[property].values.T
    a = mpl_scatter_density.ScatterDensityArtist(ax, *data, origin='low', cmap='magma')
    mappable = ax.add_artist(a)
    ax.set_xlim(*range_[0])
    ax.set_ylim(*range_[1])

    ax.set(title='Image',
           xlabel='Position_x',
           ylabel='Position_y'
           )

    if show is True:
        plt.show()

    return mappable


