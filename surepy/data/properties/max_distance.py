import numpy as np
from scipy.spatial.distance import pdist, squareform

# todo: there is a memory problem here with large files
def max_distance(locdata):
    """
    Return maximum distance between any two localizations in locdata.

    Parameters
    ----------
    locdata : LocData
        localization data

    Returns
    -------
    dict
        A dict with key being the property name and the value being
        the maximum distance between any two localizations in locdata.
    """
    D = pdist(locdata.coordinates)
    D = squareform(D)
    distance = np.nanmax(D)

    # indices of identified points (not used yet)
    # [I_row, I_col] = np.unravel_index(np.argmax(D), D.shape)

    return {'Max_distance': distance}