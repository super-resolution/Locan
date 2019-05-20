"""

This module provides functions for rendering locdata objects in 2D.

"""

import numpy as np
import matplotlib.pyplot as plt
import fast_histogram
from skimage import exposure

import surepy.data.properties.locdata_statistics


__all__ = ['render_2d']


def render_2d(locdata, ax=None, show=True, bin_size=10, range='auto', rescale=True, cmap='magma'):
    """
    Render localization data into a 2D image by binning x,y-coordinates into regular bins.

    Prepare matplotlib axes with image.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    ax : matplotlib axes
        The axes on which to show the image
    show : bool
        Flag indicating if plt.show() and the creation of a colorbar is active.
    bin_size : int or float
        x and y size of bins
    range : [[min,max],[min,max]] or 'auto' (default) or 'zero'
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
        We recommend to use one of 'viridis', 'plasma', 'magma', 'inferno', 'hot', 'hsv'.

    Returns
    -------
    matplotlib.image.AxesImage (rtype from imshow)
        mappable to create colorbar
    """

    # Provide matplotlib axes if not provided
    if not len(locdata):
        raise ValueError('Locdata does not contain any data points.')

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    # determine ranges
    if isinstance(range, str):

        if range == 'auto':
            try:
                range = np.array([[locdata.properties[x] for x in ['position_x_min', 'position_x_max']],
                                  [locdata.properties[x] for x in ['position_y_min', 'position_y_max']]])
            except KeyError:
                stats = surepy.data.properties.statistics(locdata.data[['position_x', 'position_y']],
                                                          statistic_keys=('min', 'max'))
                range = np.array([[stats['position_x_min'], stats['position_x_max']],
                                  [stats['position_y_min'], stats['position_y_max']]])

        elif range == 'zero':
            try:
                range = np.array([np.array([0, 0]),
                                  [locdata.properties[x] for x in ['position_x_max', 'position_y_max']]
                                  ]).T
            except KeyError:
                stats = surepy.data.properties.statistics(locdata.data[['position_x', 'position_y']],
                                                          statistic_keys=('max'))
                range = np.array([np.array([0, 0]),
                                  [stats['position_x_max'], stats['position_y_max']]
                                  ]).T

    else:
        range = np.array(range)

    # histogram data
    bin_number = np.ceil((range[:, 1] - range[:, 0]) / bin_size)
    bin_number = bin_number.astype(int)
    # bin_centers = np.arange(range[0], range[1], bin_size) + bin_size / 2

    data = locdata.data[['position_x', 'position_y']].values.T
    img = fast_histogram.histogram2d(*data, range=range, bins=bin_number)
    img = img.T  # to show image in the same format as scatter plot

    # contrast adjustment by equalization or rescaling
    if rescale is None or rescale is False:
        pass
    elif rescale is True:
        img = exposure.rescale_intensity(img)  # scaling to min/max of img intensities
    elif isinstance(rescale, tuple):
        minmax = (img.min(), img.max())
        rescale_abs = tuple(np.multiply(np.divide(rescale, 100), (minmax[1] - minmax[0])) + minmax[0])
        img = exposure.rescale_intensity(img, in_range=rescale_abs)
    elif rescale == 'equal':
        img = exposure.equalize_hist(img)
    else:
        raise TypeError('Set rescale to tuple, None or "equal".')

    mappable = ax.imshow(img, origin='low', extent=[*range[0], *range[1]], cmap=cmap)
    ax.set(title='Image ({:.0f} nm per bin)'.format(bin_size),
           xlabel='position_x',
           ylabel='position_y'
           )

    # show figure
    if show:
        plt.colorbar(mappable)
        plt.show()

    return mappable
