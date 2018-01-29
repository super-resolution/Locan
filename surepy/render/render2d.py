'''

This module provides methods for rendering locdata objects.

'''


def render2D(locdata, ax, bin_size=10, range='auto', rescale=(2, 98)):
    """
    Render localization data into a 2D image by binning x,y-coordinates into regular bins.

    Prepare matplotlib axes with image.

    Parameters
    ----------
    bin_size : int or float
        x and y size of bins
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
    import fast_histogram
    from skimage import exposure
    import numpy as np
    import surepy.data.properties.statistics

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
    else:
        p2, p98 = np.percentile(img, rescale)
        img = exposure.rescale_intensity(img, in_range=(p2, p98))

    # todo: colormaps: 'viridis', 'plasma', 'magma', 'inferno', 'hot', 'hsv'
    mappable = ax.imshow(img, origin='low', extent=[*range[0], *range[1]], cmap='magma')
    ax.set(title='Image (%.0f nm per bin)' % bin_size,
           xlabel='Position_x',
           ylabel='Position_y'
           )

    return mappable