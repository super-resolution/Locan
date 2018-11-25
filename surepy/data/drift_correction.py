'''

Applying drift correction on localization coordinates.

This module provides functions for drift correction of localization data.
The functions take LocData as input and compute new LocData objects.
'''

def drift_correction(selection, *args):
    """
    Transform coordinates to correct for slow drift.

    Parameters
    ----------
    selection : Selection
        specifying the localization data on which to perform the manipulation.
    args :
        transformation parameters

    Returns
    -------
    Selection
        a new instance of Selection referring to the same Dataset as the input Selection.
    """
    raise NotImplementedError