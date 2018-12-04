"""

Filter localization data.

This module provides functions for filtering LocData objects.
The functions take LocData as input and compute new LocData objects.

"""

import numpy as np
from surepy import LocData


def select_by_condition(locdata, condition):
    """
    Select by specifying conditions on data properties.

    Parameters
    ----------
    locdata : LocData
        Specifying the localization data from which to select localization data.
    condition : string
        Conditions as input in select method.
        More precise: query specifications to be used with pandas query.

    Returns
    -------
    LocData
        a new instance of LocData referring to the specified dataset.
    """

    # select
    new_indices = locdata.data.query(condition).index.values.tolist()

    # instantiate
    new_locdata = LocData.from_selection(locdata=locdata, indices=new_indices)

    # metadata
    del new_locdata.meta.history[:]
    new_locdata.meta.history.add(name='select_by_condition',
                         parameter='locdata={}, condition={}'.format(locdata, condition))

    return new_locdata


def select_by_region(locdata, roi):
    """
    Select localizations within specified rectangle, ellipse, polygon or 3D equivalents.

    Parameters
    ----------
    locdata : LocData
        Specifying the localization data from which to select localization data.
    roi : Roi Object or dict
        Region of interest as specified by Roi or dictionary with keys 'points' and 'type'. For Roi objects the
        reference attribute is ignored. Points are a list of tuples representing 1D, 2D or 3D coordinates.
        Type is a string identifier that can be either rectangle, ellipse, or polygon.

    Returns
    -------
    LocData
        a new instance of LocData referring to the specified dataset.
    """
    # todo implement ellipse and polygon for 2D and 3D
    try:
        _roi = dict(points=roi.points, type=roi.type)
    except AttributeError:
        _roi = roi

    if _roi['type']=='rectangle':
        if len(_roi['points'])==2:
            return select_by_condition(locdata, condition='{0} <= Position_x <= {1}'.format(*_roi['points']))
        if len(_roi['points'])==4:
            return select_by_condition(locdata, condition='{0} <= Position_x <= {1} and '
                                                          '{2} <= Position_y <= {3}'.format(*_roi['points']))
        if len(_roi['points'])==6:
            return select_by_condition(locdata, condition='{0} <= Position_x <= {1} and '
                                                          '{2} <= Position_y <= {3} and '
                                                          '{4} <= Position_z <= {5}'.format(*_roi['points']))

    else:
        raise NotImplementedError


def select_by_image_mask(selection, mask, pixel_size):
    """
    Select by masking using a binary image(e.g. generated by thresholding a transmitted-light microscopy image.

    Parameters
    ----------
    selection : Selection
        specifying the localization data from which to select localization data.
    mask :
        binary image.
    pixel_size : tuple(float)
        pixel sizes for each dimension in units of localization coordinates.

    Returns
    -------
    Selection
        a new instance of Selection referring to the specified dataset.
    """
    raise NotImplementedError


def exclude_sparce_points(selection, threshold_density):
    """
    Exclude localizations by thresholding a local density.

    Parameters
    ----------
    selection : Selection
        specifying the localization data on which to perform the manipulation.
    threshold_density : float
        all points with a local density below the threshold will be excluded.

    Returns
    -------
    Selection
        a new instance of Selection referring to the same Dataset as the input Selection.
    """
    raise NotImplementedError


def random_subset(locdata, number_points):
    """
    Take a random subset of localizations.

    Parameters
    ----------
    locdata : LocData
        Specifying the localization data from which to select localization data.
    number_points : int
        Number of localizations to randomly choose from locdata.

    Returns
    -------
    LocData
        a new instance of LocData carrying the subset of localizations.
    """
    indices = np.random.choice(len(locdata), size=number_points)
    new_locdata = LocData.from_selection(locdata, indices)
    new_locdata.meta.history.add(name='random subset')

    return new_locdata