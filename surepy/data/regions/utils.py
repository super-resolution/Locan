"""

Utility methods for working with regions.

"""
from pathlib import Path
import warnings

import numpy as np
from shapely.geometry import Point, MultiPoint, LineString, Polygon
from shapely.ops import cascaded_union
from shapely.prepared import prep

from surepy.data.regions.region import RoiRegion


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
