"""

Transform localization data.

This module takes localization data and applies transformation procedures on coordinates or other properties.

"""
import time

import numpy as np
import pandas as pd

from surepy import LocData
import surepy.data.hulls
from surepy.simulation import simulate_csr
from surepy.data import metadata_pb2


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
        New localization data with tansformed coordinates.
    """
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
        # update metadata
        meta_ = metadata_pb2.Metadata()
        meta_.CopyFrom(locdata.meta)
        try:
            meta_.ClearField("identifier")
        except ValueError:
            pass

        try:
            meta_.ClearField("element_count")
        except ValueError:
            pass

        try:
            meta_.ClearField("frame_count")
        except ValueError:
            pass

        meta_.modification_date = int(time.time())
        meta_.state = metadata_pb2.MODIFIED
        meta_.ancestor_identifiers.append(locdata.meta.identifier)
        meta_.history.add(name='transform_affine', parameter=f'matrix={matrix}, offset={offset}')

        # new LocData object
        new_locdata = LocData.from_dataframe(pd.DataFrame(transformed_points, columns=locdata.coordinate_labels),
                                             meta=meta_)
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

