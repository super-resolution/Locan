"""

Filter localization data.

This module provides functions for filtering LocData objects.
The functions take LocData as input and compute new LocData objects.

"""
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.path as mpath

from surepy import LocData
from surepy.constants import N_JOBS
from surepy.data.metadata_utils import _modify_meta
from surepy.data.rois import RoiRegion


def select_by_condition(locdata, condition):
    """
    Select by specifying conditions on data properties.

    Parameters
    ----------
    locdata : LocData
        Specifying the localization data from which to select localization data.
    condition : string
        Conditions as input in select method.
        More precise: query specifications to be used with pandas query.

    Returns
    -------
    LocData
        a new instance of LocData referring to the specified dataset.
    """
    local_parameter = locals()

    # select
    new_indices = locdata.data.query(condition).index.values.tolist()

    # instantiate
    new_locdata = LocData.from_selection(locdata=locdata, indices=new_indices)

    # update metadata
    meta_ = _modify_meta(locdata, function_name=sys._getframe().f_code.co_name, parameter=local_parameter, meta=None)
    new_locdata.meta = meta_

    return new_locdata


def _select_by_region(locdata, roi, reduce=True):
    """
    Select localizations within specified region. Region can be rectangle, ellipse, polygon or 3D equivalents.

    Parameters
    ----------
    locdata : LocData
        Specifying the localization data from which to select localization data.
    roi : Roi Object or dict
        Region of interest as specified by Roi or dictionary with keys 'region_specs' and 'type'. For Roi objects the
        reference attribute is ignored. Allowed values for `region_specs` and `type` are defined for Roi objects.
    reduce : Bool
        Return the reduced LocData object or keep references alive.

    Returns
    -------
    LocData
        New instance of LocData referring to the specified dataset.
    """
    # todo implement ellipse and polygon for 2D and 3D
    try:
        roi_ = dict(region_specs=roi.region_specs, type=roi.type)
    except AttributeError:
        roi_ = roi

    if roi_['type'] == 'rectangle' and roi_['region_specs'][-1] == 0:
        min_x, min_y = roi_['region_specs'][0]
        max_x, max_y = list((a + b for a, b in zip(roi_['region_specs'][0], roi_['region_specs'][1:3])))
        new_locdata = select_by_condition(locdata, condition=f'{min_x} <= Position_x <= {max_x} and '
                                            f'{min_y} <= Position_y <= {max_y}')

    if roi_['type'] == 'ellipse':
        pass

    if roi_['type'] == 'polygon':
        polygon = roi_['region_specs']
        path = mpath.Path(polygon)
        inside = path.contains_points(locdata.coordinates)
        new_indices = np.where(inside)
        new_locdata = LocData.from_selection(locdata=locdata, indices=new_indices)

    else:
        raise NotImplementedError




    # finish
    if reduce:
        new_locdata.reduce()

    # meta is updated by select_by_condition function. No further updates needed.
    return new_locdata

def select_by_region(locdata, region, reduce=True):
    """
    Select localizations within specified region of interest.

    Parameters
    ----------
    locdata : LocData
        Specifying the localization data from which to select localization data.
    region : RoiRegion Object, or dict
        Region of interest as specified by RoiRegion or dictionary with keys `region_specs` and `region_type`.
        Allowed values for `region_specs` and `region_type` are defined in the docstrings for `Roi` and `RoiRegion`.
    reduce : Bool
        Return the reduced LocData object or keep references alive.

    Returns
    -------
    LocData
        New instance of LocData referring to the specified dataset.
    """
    if isinstance(region, dict):
        roi = RoiRegion(region_specs=region.region_specs, type=region.type)
    except AttributeError:
        roi_ = region

    if roi_['type'] == 'rectangle' and roi_['region_specs'][-1] == 0:
        min_x, min_y = roi_['region_specs'][0]
        max_x, max_y = list((a + b for a, b in zip(roi_['region_specs'][0], roi_['region_specs'][1:3])))
        new_locdata = select_by_condition(locdata, condition=f'{min_x} <= Position_x <= {max_x} and '
                                            f'{min_y} <= Position_y <= {max_y}')

    if roi_['type'] == 'ellipse':
        pass

    if roi_['type'] == 'polygon':
        polygon = roi_['region_specs']
        path = mpath.Path(polygon)
        inside = path.contains_points(locdata.coordinates)
        new_indices = np.where(inside)
        new_locdata = LocData.from_selection(locdata=locdata, indices=new_indices)

    else:
        raise NotImplementedError




    # finish
    if reduce:
        new_locdata.reduce()

    # meta is updated by select_by_condition function. No further updates needed.
    return new_locdata

def select_by_image_mask(selection, mask, pixel_size):
    """
    Select by masking using a binary image(e.g. generated by thresholding a transmitted-light microscopy image.

    Parameters
    ----------
    selection : Selection
        specifying the localization data from which to select localization data.
    mask :
        binary image.
    pixel_size : tuple(float)
        pixel sizes for each dimension in units of localization coordinates.

    Returns
    -------
    Selection
        a new instance of Selection referring to the specified dataset.
    """
    raise NotImplementedError


def exclude_sparse_points(locdata, other_locdata=None, radius=50, min_samples=5):
    """
    Exclude localizations by thresholding a local density.

    A subset of localizations, that exhibit a small local density of localizations from locdata or alternatively from
    other_locdata, is identified as noise and excluded.
    Noise is identified by using a nearest-neighbor search (sklearn.neighbors.NearestNeighbors) to find all
    localizations within a circle (sphere) of the given `radius`. If the number of localizations is below the
    threshold value `min_samples`, the localization is considered to be noise.

    The method identifies the same noise points as done by the clustering algorithm DBSCAN [1]_.

    Parameters
    ----------
    locdata : LocData
        Specifying the localization data from which to exclude localization data.
    other_locdata : LocData
        Specifying the localization data on which to compute local density.
    radius: float
        Radius of a circle or sphere in which neighbors are identified (equivalent to epsilon in DBSCAN).
    min_samples : int
        The minimum number of samples in the neighborhood that need to be found for each localization to not be
        identified as noise (equivalent to minPoints in DBSCAN).

    Returns
    -------
    Tuple of LocData
        Two Locdata objects are returned, one carrying all noise localizations, the other carrying all localizations
        except noise.

    References
    ----------
    .. [1] Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu,
       A density-based algorithm for discovering clusters in large spatial databases with noise.
       In: Evangelos Simoudis, Jiawei Han, Usama M. Fayyad (Hrsg.): Proceedings of the Second International Conference
       on Knowledge Discovery and Data Mining (KDD-96). AAAI Press, 1996, S. 226-231, ISBN 1-57735-004-9.
    """
    local_parameter = locals()

    if other_locdata is None:
        nn = NearestNeighbors(metric='euclidean', n_jobs=N_JOBS).fit(locdata.coordinates)
        neighbor_points_list = nn.radius_neighbors(radius=radius, return_distance=False)
        # if points is not provided the query point is not considered its own neighbor.
    else:
        nn = NearestNeighbors(metric='euclidean', n_jobs=N_JOBS).fit(other_locdata.coordinates)
        neighbor_points_list = nn.radius_neighbors(locdata.coordinates, radius=radius, return_distance=False)

    indices_to_keep = [len(pts) >= min_samples for pts in neighbor_points_list]
    new_locdata = LocData.from_selection(locdata, indices_to_keep)

    # update metadata
    meta_ = _modify_meta(locdata, function_name=sys._getframe().f_code.co_name, parameter=local_parameter, meta=None)
    new_locdata.meta = meta_

    return new_locdata


def random_subset(locdata, number_points):
    """
    Take a random subset of localizations.

    Parameters
    ----------
    locdata : LocData
        Specifying the localization data from which to select localization data.
    number_points : int
        Number of localizations to randomly choose from locdata.

    Returns
    -------
    LocData
        a new instance of LocData carrying the subset of localizations.
    """
    local_parameter = locals()

    indices = np.random.choice(len(locdata), size=number_points)
    new_locdata = LocData.from_selection(locdata, indices)

    # update metadata
    meta_ = _modify_meta(locdata, function_name=sys._getframe().f_code.co_name, parameter=local_parameter, meta=None)
    new_locdata = LocData.from_selection(locdata, indices, meta=meta_)

    return new_locdata
