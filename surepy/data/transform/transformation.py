"""

Transform localization data.

This module takes localization data and applies transformation procedures on coordinates or other properties.

"""

import numpy as np
import pandas as pd

from surepy import LocData
import surepy.data.hulls
from surepy.simulation import simulate_csr


def transform(locdata, *args):
    """
    Transform coordinates for correcting various aberrations.

    Parameters
    ----------
    locdata : LocData object
        Localization data on which to perform the manipulation.
    args :
        transformation parameters

    Returns
    -------
    locdata : LocData object
        New localization data with tansformed coordinates.

    Notes
    -----
    Not yet implemented.
    """
    raise NotImplementedError


def randomize(locdata, hull_region='bb'):
    """
    Transform locdata coordinates into randomized coordinates that follow complete spatial randomness on the same
    region as the input locdata.

    Parameters
    ----------
    locdata : LocData object
        Localization data to be randomized
    hull_region : str
        One of 'bb', 'ch', 'as', 'obb' referring to the corresponding hull.

    Returns
    -------
    locdata : LocData object
        New localization data with randomized coordinates.
    """

    if hull_region is 'bb':
        try:
            ranges = locdata.bounding_box.hull.T
        except AttributeError:
            locdata.bounding_box = surepy.data.hulls.Bounding_box(locdata.coordinates)
            ranges = locdata.bounding_box.hull.T

        if len(ranges)==2:
            new_locdata = simulate_csr(n_samples=len(locdata), x_range=ranges[0], y_range=ranges[1])
        elif len(ranges)==3:
            new_locdata = simulate_csr(n_samples=len(locdata), x_range=ranges[0], y_range=ranges[1], z_range=ranges[2])
        else:
            raise TypeError()

    # todo: implement simulate_csr on various polygon regions
    # if hull_region is 'ch':
    #
    #     n = len(locdata)
    #     dim = len(locdata.coordinate_labels)
    #
    #     try:
    #         min_coordinates = locdata.convex_hull.hull[0]
    #     except AttributeError:
    #         locdata.bounding_box = surepy.data.hulls.Bounding_box(locdata.coordinates)
    #         min_coordinates = locdata.bounding_box.hull[0]
    #
    #     width = locdata.bounding_box.width
    #     new_coordinates = np.random.random((n, dim)) * width + min_coordinates
    #
    #     new_locdata = simulate_csr(n_samples=len(locdata), x_range=)

    else:
        raise NotImplementedError

    return new_locdata

