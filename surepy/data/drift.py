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
from itertools import accumulate

import numpy as np
import pandas as pd

try:
    import open3d as o3d
    _has_open3d = True
except ImportError:
    _has_open3d = False

from surepy.data.locdata import LocData
from surepy.data.register import _register_icp_open3d
from surepy.data.transform.transformation import transform_affine


__all__ = ['drift_correction']


def drift_correction(locdata, chunk_size=1000, target='first'):
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

    Returns
    -------
    LocData
        a new instance of LocData referring to the input `locdata`.
    """
    if not _has_open3d:
        raise ImportError("open3d is required.")

    local_parameter = locals()

    # split in chunks
    chunk_sizes = [chunk_size] * (len(locdata) // chunk_size) + [(len(locdata) % chunk_size)]
    cum_chunk_sizes = list(accumulate(chunk_sizes))
    cum_chunk_sizes.insert(0, 0)
    index_lists = [locdata.data.index[slice(lower, upper)]
                   for lower, upper in zip(cum_chunk_sizes[:-1], cum_chunk_sizes[1:])]
    locdatas = [LocData.from_selection(locdata=locdata, indices=index_list) for index_list in index_lists]
    # return locdatas

    # register and transform locdata
    matrices = []
    offsets = []
    transformed_locdatas = []
    if target is 'first':
        for locdata in locdatas[1:]:
            matrix, offset = _register_icp_open3d(locdata.coordinates, locdatas[0].coordinates,
                                                  matrix=None, offset=None, pre_translation=None,
                                                  max_correspondence_distance=100, max_iteration=10_000,
                                                  verbose=False)
            matrices.append(matrix)
            offsets.append(offset)

        transformed_locdatas = [transform_affine(locdata, matrix, offset)
                                for locdata, matrix, offset in zip(locdatas[1:], matrices, offsets)]

    elif target is 'previous':
        for n in range(len(locdatas)-1):
            matrix, offset = _register_icp_open3d(locdatas[n+1].coordinates, locdatas[n].coordinates,
                                                  matrix=None, offset=None, pre_translation=None,
                                                  max_correspondence_distance=100, max_iteration=10_000,
                                                  with_scaling=False, verbose=False)
            matrices.append(matrix)
            offsets.append(offset)

        for n, locdata in enumerate(locdatas[1:]):
            transformed_locdata = locdata
            for matrix, offset in zip(reversed(matrices[:n]), reversed(offsets[:n])):
                transformed_locdata = transform_affine(transformed_locdata, matrix, offset)
            transformed_locdatas.append(transformed_locdata)
    # return matrices, offsets

    new_locdata = LocData.concat([locdatas[0]] + transformed_locdatas)
    return new_locdata, matrices, offsets

# todo: class Drift carrying the drift estimate. Show drift as function of frame and fit continuous function.
#  Plus function to apply drift correction.
