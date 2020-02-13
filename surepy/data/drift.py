"""

Drift correction for localization coordinates.

This module provides functions for applying drift correction of localization data.
The functions take LocData as input and compute new LocData objects.

Software based drift correction has been described in several publications [1]_, [2]_, [3]_.
Methods employed for drift estimation comprise single molecule localization analysis or image correlation analysis.

Note
----
The procedure is in an exploratory state and has not been fully developed and tested.

References
----------
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
from surepy.data.locdata import LocData
from surepy.analysis.drift import Drift
from surepy.data.transform.transformation import transform_affine
from surepy.constants import _has_open3d
if _has_open3d: import open3d as o3d


__all__ = ['drift_correction']


def drift_correction(locdata, chunk_size=1000, target='first', analysis_class=None):
    """
    Transform coordinates to correct for slow drift by registering points in successive time-chunks of localization
    data using an "Iterative Closest Point" algorithm.

    Parameters
    ----------
    locdata : LocData object
        Localization data representing the source on which to perform the manipulation.
    chunk_size : int
        Number of consecutive localizations to form a single chunk of data.
    target : string
        The chunk on which all other chunks are aligned. One of 'first', 'previous'.
    analysis_class : Drift object
        Precomputed Drift (Analysis) class

    Returns
    -------
    LocData
        a new instance of LocData referring to the input `locdata`.
    """
    if not _has_open3d:
        raise ImportError("open3d is required.")

    local_parameter = locals()

    if analysis_class is None:
        drift = Drift(chunk_size=chunk_size, target=target).compute(locdata)
    else:
        drift = analysis_class

    transformed_locdatas = []
    if target == 'first':
        transformed_locdatas = [transform_affine(locdata, matrix, offset) for locdata, matrix, offset
                                in zip(drift.collection.references[1:], drift.results.matrices, drift.results.offsets)]
    elif target == 'previous':
        for n, locdata in enumerate(drift.collection.references[1:]):
            transformed_locdata = locdata
            for matrix, offset in zip(reversed(drift.results.matrices[:n]), reversed(drift.results.offsets[:n])):
                transformed_locdata = transform_affine(transformed_locdata, matrix, offset)
            transformed_locdatas.append(transformed_locdata)

    new_locdata = LocData.concat([drift.collection.references[0]] + transformed_locdatas)

    # todo: add meta data
    return new_locdata
