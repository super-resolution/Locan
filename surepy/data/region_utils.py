"""

Utility functions for working with regions.

"""

from shapely.geometry import Polygon
from shapely.ops import cascaded_union

from surepy.data.locdata import LocData
# import surepy.data.locdata  # cannot import LocData directly due to convoluted circular import issues
from surepy.constants import HullType
from surepy.data.region import RoiRegion
import surepy.data.rois


__all__ = ['surrounding_region']


def surrounding_region(region, distance=100, support=None):
    """
    Define surrounding region by extending a region and returning the extended region excluding the input region.
    If region contains a list of regions, the unification of all extended regions is returned.

    Parameters
    ----------
    region : RoiRegion, list of RoiRegion or shapely Polygon
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


def localizations_in_cluster_regions(locdata, collection, hull_type=HullType.CONVEX_HULL):
    """
    Identify localizations from `locdata` within the regions of all `collection` elements.

    Parameters
    ----------
    locdata : LocData
        Localization data that is tested for being inside the region

    collection : LocData or list(LocData)
        A set of Locdata objects collected in a collection or list.

    hull_type : surepy.HullType
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
