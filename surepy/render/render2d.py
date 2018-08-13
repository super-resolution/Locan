'''

This module provides methods for rendering locdata objects.

'''

import numpy as np
import matplotlib.pyplot as plt
import fast_histogram
from skimage import exposure

import surepy.data.properties.statistics


def render2D(locdata, ax=None, show=True, bin_size=10, range='auto', rescale=(2, 98), cmap='magma'):
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
        Flag indicating if plt.show() is active.
    bin_size : int or float
        x and y size of bins
    range : [[min,max],[min,max]] or 'auto' or 'zero'
        defining the binned region by [min, max] ranges from input;
        for 'auto' by [min, max] ranges from data; for 'zero' by [0, max] ranges from data.
    rescale : tuple, None, or 'equal'
        rescale intensity values to be within percentile (tuple with upper and lower bounds provided in percent).
        For None intensity values are rescaled to the min and max possible values of the given representation.
        For 'equal' intensity values are rescaled by histogram equalization.
    cmap : str or Colormap instance
        The colormap used to map normalized data values to RGBA colors.
        We recommend to use one of 'viridis', 'plasma', 'magma', 'inferno', 'hot', 'hsv'.


    Returns
    -------
    matplotlib.image.AxesImage (rtype from imshow)
        mappable to create colorbar

    """

    # Provide matplotlib axes if not provided
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    # determine ranges
    if isinstance(range, str):

        if range == 'auto':
            try:
                range = np.array([[locdata.properties[x] for x in ['Position_x_min', 'Position_x_max']],
                                  [locdata.properties[x] for x in ['Position_y_min', 'Position_y_max']]])
            except KeyError:
                stats = surepy.data.properties.statistics(locdata.data[['Position_x', 'Position_y']],
                                                          statistic_keys=('min', 'max'))
                range = np.array([[stats['Position_x_min'], stats['Position_x_max']],
                                  [stats['Position_y_min'], stats['Position_y_max']]])

        elif range == 'zero':


            try:
                range = np.array([np.array([0, 0]),
                                   [locdata.properties[x] for x in ['Position_x_max', 'Position_y_max']]]
                                  ).T
            except KeyError:
                stats = surepy.data.properties.statistics(locdata.data[['Position_x', 'Position_y']],
                                                          statistic_keys=('max'))
                range = np.array([np.array([0, 0]),
                                  [stats['Position_x_max'], stats['Position_y_max']]]
                                 ).T

    else:
        range = np.array(range)

    # histogram data
    bin_number = np.ceil((range[:, 1] - range[:, 0]) / bin_size)
    bin_number = bin_number.astype(int)
    # bin_centers = np.arange(range[0], range[1], bin_size) + bin_size / 2

    data = locdata.data[['Position_x', 'Position_y']].values.T
    img = fast_histogram.histogram2d(*data, range=range, bins=bin_number)
    img = img.T  # to show image in the same format as scatter plot

    # contrast adjustment by equalization or rescaling
    if isinstance(rescale, str):
        if rescale == 'equal':
            img = exposure.equalize_hist(img)
    elif rescale is None:
        img = exposure.rescale_intensity(img)
    else:
        p_low, p_up = np.percentile(img, rescale)
        img = exposure.rescale_intensity(img, in_range=(p_low, p_up))

    mappable = ax.imshow(img, origin='low', extent=[*range[0], *range[1]], cmap=cmap)
    ax.set(title='Image (%.0f nm per bin)' % bin_size,
           xlabel='Position_x',
           ylabel='Position_y'
           )

    # show figure
    if show:
        plt.colorbar(mappable)
        plt.show()

    return mappable