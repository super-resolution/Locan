"""
Utility functions for binning and rendering in 2 and 3 dimensions.

# todo: has been moved; remove whole file
"""
import numpy as np


__all__ = []


# todo: deprecate
def _coordinate_ranges(locdata, range=None):
    """
    Provide coordinate bin_range for locdata that can be fed into a binning algorithm.

    Parameters
    ----------
    locdata : pandas DataFrame or LocData object
        Localization data.
    range : tuple with shape (dimension, 2) or None or str 'zero'
        ((min_x, max_x), (min_y, max_y), ...) bin_range for each coordinate;
        for None (min, max) bin_range are determined from data;
        for 'zero' (0, max) bin_range with max determined from data.

    Returns
    -------
    numpy array of float with shape (dimension, 2)
        A bin_range (min, max) for each available coordinate.
    """
    if range is None:
        ranges_ = locdata.bounding_box.hull.T
    elif isinstance(range, str):
        if range == 'zero':
            ranges_ = locdata.bounding_box.hull
            ranges_[0] = np.zeros(len(ranges_))
            ranges_ = ranges_.T
        else:
            raise ValueError(f'The parameter bin_range={range} is not defined.')
    else:
        if np.ndim(range) != locdata.dimension:
            raise TypeError(f'The tuple {range} must have the same dimension as locdata which is {locdata.dimension}.')
        else:
            ranges_ = np.asarray(range)

    return ranges_
