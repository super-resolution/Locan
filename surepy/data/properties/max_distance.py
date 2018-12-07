"""
Compute maximum distance.

The maximum distance between any two localizations in locdata is computed.
This value represents a new property of locdata.

"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

# todo: there is a memory problem here with large files
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
        A dict with key `Max_distance` and the corresponding value being the maximum distance.
    """
    D = pdist(locdata.coordinates)
    D = squareform(D)
    distance = np.nanmax(D)

    # indices of identified points (not used yet)
    # [I_row, I_col] = np.unravel_index(np.argmax(D), D.shape)

    return {'Max_distance': distance}