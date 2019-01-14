"""

Drift correction for localization coordinates.

This module provides functions for applying drift correction of localization data.
The functions take LocData as input and compute new LocData objects.

Software based drift correction has been described in several publications [1]_, [2]_, [3]_.
Methods employed for drift estimation comprise single molecule localization analysis or image correlation analysis.

.. [1] C. Geisler,
   Drift estimation for single marker switching based imaging schemes,
   Optics Express. 2012, 20(7):7274-89.

.. [2] Yina Wang et al.,
   Localization events-based sample drift correction for localization microscopy with redundant cross-correlation
   algorithm, Optics Express 2014, 22(13):15982-91.

.. [3] Michael J. Mlodzianoski et al.,
   Sample drift correction in 3D fluorescence photoactivation localization microscopy,
   Opt Express. 2011 Aug 1;19(16):15009-19.

"""

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