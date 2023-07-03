"""

Utility functions for working with regions.

See Also
--------
:func:`locan.data.filter.select_by_region`
:func:`locan.data.properties.misc.distance_to_region`
:func:`locan.data.properties.misc.distance_to_region_boundary`
"""
from __future__ import annotations

from shapely.ops import unary_union

from locan.data.region import EmptyRegion, Region2D, RoiRegion

__all__: list[str] = ["regions_union", "expand_region", "surrounding_region"]


def regions_union(regions):
    """
    Return the union of `regions`.

    Parameters
    ----------
    regions : list[Region]
        Original region(s)

    Returns
    --------
    Region
    """
    if all([isinstance(region, (Region2D, RoiRegion)) for region in regions]):
        shapely_objects = [reg.shapely_object for reg in regions]
        unified_regions = unary_union(shapely_objects)
        if unified_regions.is_empty:
            return EmptyRegion()
        else:
            return Region2D.from_shapely(unified_regions)
    else:
        raise NotImplementedError("regions must all be Region2D")


def expand_region(region, distance=100, support=None, **kwargs):
    """
    Expand a region by `distance`.
    If region contains a list of regions, the unification of all expanded regions is
    returned.

    Parameters
    ----------
    region : Region | shapely.Polygon
        Original region(s)
    distance : int | float
        Distance by which the region is expanded orthogonal to its boundary.
    support : Region | None
        A region defining the maximum outer boundary.
    kwargs : dict
        Other parameters passed to :func:`shapely.geometry.buffer`
        for :class:`Region2D` objects.

    Returns
    --------
    Region
    """
    expanded_region = region.buffer(distance, **kwargs)

    if support is not None:
        expanded_region = support.intersection(expanded_region)

    try:
        return Region2D.from_shapely(expanded_region)
    except AttributeError:
        return expanded_region


def surrounding_region(region, distance=100, support=None, **kwargs):
    """
    Define surrounding region by extending a region and returning the extended
    region excluding the input region.
    If region contains a list of regions, the unification of all extended
    regions is returned.

    Parameters
    ----------
    region : Region
        Original region(s)
    distance : int | float
        Distance by which the region is extended orthogonal to its boundary.
    support : Region | None
        A region defining the maximum outer boundary.
    kwargs : dict
        Other parameters passed to :func:`shapely.geometry.buffer` for
        :class:`Region2D` objects.

    Returns
    --------
    Region
    """
    extended_region = expand_region(
        region, distance=distance, support=support, **kwargs
    )
    if isinstance(extended_region, Region2D):
        surrounding_region_ = extended_region.symmetric_difference(region)
        return Region2D.from_shapely(surrounding_region_)
    else:
        raise NotImplementedError("Only 2-dimensional function has been implemented.")
