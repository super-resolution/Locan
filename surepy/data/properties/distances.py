"""
Compute maximum distance.

The maximum distance between any two localizations in locdata is computed.
This value represents a new property of locdata.

"""

import numpy as np
from scipy.spatial.distance import pdist


__all__ = ['max_distance']


def max_distance(locdata):
    """
    Return maximum distance between any two localizations in locdata.

    Parameters
    ----------
    locdata : LocData
        Localization data

    Returns
    -------
    dict
        A dict with key `max_distance` and the corresponding value being the maximum distance.
    """
    points = locdata.convex_hull.vertices
    D = pdist(points)
    distance = np.nanmax(D)
    return {'max_distance': distance}
