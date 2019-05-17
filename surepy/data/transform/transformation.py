"""

Transform localization data.

This module takes localization data and applies transformation procedures on coordinates or other properties.

"""
import sys
import time

import numpy as np
import pandas as pd

from surepy.data.locdata import LocData
import surepy.data.hulls
from surepy.data.region import RoiRegion
from surepy.simulation import simulate_csr, simulate_csr_on_region
from surepy.data.metadata_utils import _modify_meta


__all__ = ['transform_affine', 'randomize']


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


def transform_affine(locdata, matrix=None, offset=None):
    """
    Transform `points` or coordinates in `locdata` by an affine transformation.

    Parameters
    ----------
    locdata : ndarray or LocData object
        Localization data on which to perform the manipulation.
    matrix :
        Transformation matrix.
    offset : tuple of int or float
        Values for translation.

    Returns
    -------
    locdata : ndarray or LocData object
        New localization data with transformed coordinates.
    """
    local_parameter = locals()

    if len(locdata) == 0:
        return locdata

    # adjust input
    if isinstance(locdata, LocData):
        points = locdata.coordinates
    else:
        points = locdata

    # transform
    if matrix is None:
        m = np.identity(len(points[0]))
    else:
        m = matrix
    if offset is None:
        o = np.zeros(len(points[0]))
    else:
        o = offset

    transformed_points = np.array([np.dot(m, point) + o for point in points])

    # prepare output
    if isinstance(locdata, LocData):
        # new LocData object
        new_dataframe = locdata.data.copy()
        new_dataframe.update(pd.DataFrame(transformed_points, columns=locdata.coordinate_labels))
        new_locdata = LocData.from_dataframe(new_dataframe)

        # update metadata
        meta_ = _modify_meta(locdata, new_locdata, function_name=sys._getframe().f_code.co_name,
                             parameter=local_parameter,
                             meta=None)
        new_locdata.meta = meta_

        return new_locdata

    else:
        return transformed_points


def randomize(locdata, hull_region='bb'):
    """
    Transform locdata coordinates into randomized coordinates that follow complete spatial randomness on the same
    region as the input locdata.

    Parameters
    ----------
    locdata : LocData object
        Localization data to be randomized
    hull_region : str, RoiRegion Object, or dict
        Region of interest as specified by a hull or a `RoiRegion` or dictionary with keys `region_specs` and
        `region_type`.
        Allowed values for `region_specs` and `region_type` are defined in the docstrings for `Roi` and `RoiRegion`.
        String identifier can be one of 'bb', 'ch', 'as', 'obb' referring to the corresponding hull.

    Returns
    -------
    locdata : LocData object
        New localization data with randomized coordinates.
    """
    local_parameter = locals()

    if hull_region is 'bb':
        try:
            ranges = locdata.bounding_box.hull.T
        except AttributeError:
            locdata.bounding_box = surepy.data.hulls.BoundingBox(locdata.coordinates)
            ranges = locdata.bounding_box.hull.T

        new_locdata = simulate_csr(n_samples=len(locdata), n_features=len(ranges), feature_range=ranges)

    elif hull_region is 'ch':
        try:
            region_ = locdata.convex_hull.region
        except AttributeError:
            region_ = surepy.data.hulls.ConvexHull(locdata.coordinates).region

        new_locdata = simulate_csr_on_region(region_, n_samples=len(locdata))

    # todo: implement simulate_csr on as and obb hull regions

    elif isinstance(hull_region, (RoiRegion, dict)):
        if isinstance(hull_region, dict):
            region_ = RoiRegion(region_specs=hull_region['region_specs'], region_type=hull_region['region_type'])
        else:
            region_ = hull_region

        new_locdata = simulate_csr_on_region(region_, n_samples=len(locdata))

    else:
        raise NotImplementedError

    # update metadata
    meta_ = _modify_meta(locdata, new_locdata, function_name=sys._getframe().f_code.co_name,
                         parameter=local_parameter, meta=None)
    new_locdata.meta = meta_

    return new_locdata
