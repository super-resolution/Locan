"""

Utility functions for working with regions.

"""
import sys

import numpy as np
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import cascaded_union
from shapely.prepared import prep

from surepy.data.locdata import LocData
from surepy.data.metadata_utils import _modify_meta
from surepy.constants import HullType
from surepy.data.region import RoiRegion
import surepy.data.rois


__all__ = ['surrounding_region', 'localizations_in_region', 'localizations_in_cluster_regions', 'distance_to_region',
           'distance_to_region_boundary']


def surrounding_region(region, distance=100, support=None):
    """
    Define surrounding region by extending a region and returning the extended region excluding the input region.
    If region contains a list of regions, the unification of all extended regions is returned.

    Parameters
    ----------
    region : RoiRegion, list of RoiRegion or shapely.Polygon
        Original region(s)

    distance : int or float
        Distance by which the region is extended orthogonal to its boundary.

    support : RoiRegion or None
        A region defining the maximum outer boundary.

    Returns
    --------
    shapely.Polygon
    """
    if isinstance(region, Polygon):
        polygon = region
        extended_region = polygon.buffer(distance).difference(polygon)

    elif isinstance(region, RoiRegion):
        polygon = Polygon(region.polygon[:-1])
        extended_region = polygon.buffer(distance).difference(polygon)

    elif isinstance(region, (list, tuple)):
        polygons = []
        extended_polygons = []
        for reg in region:
            polygon = Polygon(reg.polygon[:-1])  # the polygon attribute repeats the first point in the last position
            polygons.append(polygon)
            extended_polygons.append(polygon.buffer(distance))

        unified_regions = cascaded_union(polygons)
        unified_extended_region = cascaded_union(extended_polygons)
        extended_region = unified_extended_region.difference(unified_regions)

    else:
        raise TypeError(f'The region must be a RoiRegion or list thereof, or a shapely polygon.')

    if support is not None:
        support_polygon = Polygon(support.polygon[:-1])
        extended_region = support_polygon.intersection(extended_region)

    return extended_region


def localizations_in_region(locdata, region, loc_properties=None, reduce=True):
    """
    Select localizations from `locdata` that are within `region` and return a new LocData object.
    Selection is with respect to loc_properties or to localization coordinates that correspond to region dimension.

    Parameters
    ----------
    locdata : LocData
        Localization data that is tested for being inside the region.
    region : RoiRegion or shapely.Polygon
        Region
    loc_properties : list of string or None
        Localization properties to be tested.
    reduce : Bool
        Return the reduced LocData object or keep references alive.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations within region of interest.

    Note
    ----
    Points on boundary of regions are considered outside if region is a shapely object but inside if region is a
    matplotlib (which is the case for RoiRegion) object.
    """
    local_parameter = locals()

    if isinstance(region, Polygon):
        dimension = 2
    else:
        dimension = region.dimension

    if loc_properties is None:
        loc_properties_ = locdata.coordinate_labels[0:dimension]
    else:
        loc_properties_ = loc_properties

    points = locdata.data[list(loc_properties_)].values

    if isinstance(region, Polygon):
        points = MultiPoint(points)
        prepared_polygon = prep(region)
        mask = list(map(prepared_polygon.contains, points))
        locdata_indices_to_keep = locdata.data.index[mask]
        new_locdata = LocData.from_selection(locdata=locdata, indices=locdata_indices_to_keep)
        new_locdata.region = RoiRegion(region_type='polygon', region_specs=region.exterior.coords)

    else:
        indices_inside = region.contains(points)
        locdata_indices_to_keep = locdata.data.index[indices_inside]
        new_locdata = LocData.from_selection(locdata=locdata, indices=locdata_indices_to_keep)
        new_locdata.region = region

    # finish
    if reduce:
        new_locdata.reduce()

    # update metadata
    meta_ = _modify_meta(locdata, new_locdata, function_name=sys._getframe().f_code.co_name,
                         parameter=local_parameter,
                         meta=None)
    new_locdata.meta = meta_

    return new_locdata


def localizations_in_cluster_regions(locdata, collection, hull_type=HullType.CONVEX_HULL):
    """
    Identify localizations from `locdata` within the regions of all `collection` elements.

    Parameters
    ----------
    locdata : LocData
        Localization data that is tested for being inside the region

    collection : LocData or list(LocData)
        A set of Locdata objects collected in a collection or list.

    hull_type : HullType
        The hull type for each LocData object that is used to define the region.

    Returns
    --------
    LocData
        A collection of LocData objects with all elements of locdata contained by the region.
    """
    locdatas = []
    if isinstance(collection, LocData):
        if isinstance(collection.references, list):  # this case covers pure collections
            for ref in collection.references:
                cregion = getattr(ref, hull_type.value).region
                roi = surepy.data.rois.Roi(cregion.region_type, cregion.region_specs)
                roi.reference = locdata
                locdatas.append(roi.locdata())
        else:  # this case covers selections of collections
            for index in collection.indices:
                cregion = getattr(collection.references.references[index], hull_type.value).region
                roi = surepy.data.rois.Roi(cregion.region_type, cregion.region_specs)
                roi.reference = locdata
                locdatas.append(roi.locdata())
    else:  # this case covers list of LocData objects
        for ref in collection:
            cregion = getattr(ref, hull_type.value).region
            roi = surepy.data.rois.Roi(cregion.region_type, cregion.region_specs)
            roi.reference = locdata
            locdatas.append(roi.locdata())

    new_collection = surepy.data.locdata.LocData.from_collection(locdatas)

    return new_collection


def distance_to_region(locdata, region):
    """
    Determine the distance to the nearest point within `region` for all localizations.
    Returns zero if localization is within the region.

    Parameters
    ----------
    locdata : LocData
        Localizations for which distances are determined.

    region : RoiRegion or list of RoiRegion
        Region taken from RoiRegion or the unified RoiRegions that sets the boundary.

    Returns
    --------
    ndarray
        Distance for each localization.
    """
    if isinstance(region, list):
        polygons = []
        for reg in region:
            polygon = Polygon(reg.polygon[:-1])  # the polygon attribute repeats the first point in the last position
            polygons.append(polygon)
        region_polygon = cascaded_union(polygons)

    else:
        region_polygon = Polygon(region.polygon[:-1])

    distances = np.full(len(locdata), 0.)
    for i, point in enumerate(locdata.coordinates):
        distances[i] = Point(point).distance(region_polygon)

    return distances


def distance_to_region_boundary(locdata, region):
    """
    Determine the distance to the nearest region boundary for all localizations.
    Returns a positive value regardless of wether the point is within or ouside the region.

    Parameters
    ----------
    locdata : LocData
        Localizations for which distances are determined.

    region : RoiRegion or list of RoiRegion
        Region taken from RoiRegion or the unified RoiRegions that sets the boundary.

    Returns
    --------
    ndarray
        Distance for each localization.
    """
    if isinstance(region, list):
        polygons = []
        for reg in region:
            polygon = Polygon(reg.polygon[:-1])  # the polygon attribute repeats the first point in the last position
            polygons.append(polygon)
        region_polygon = cascaded_union(polygons)

    else:
        region_polygon = Polygon(region.polygon[:-1])

    distances = np.full(len(locdata), 0.)
    for i, point in enumerate(locdata.coordinates):
        distances[i] = Point(point).distance(region_polygon.boundary)

    return distances
