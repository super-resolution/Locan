"""

Utility functions for working with regions.

See Also
--------
:func:`locan.process.filter.select_by_region`
:func:`locan.process.properties.misc.distance_to_region`
:func:`locan.process.properties.misc.distance_to_region_boundary`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from shapely.ops import unary_union

from locan.data.regions.region import (
    AxisOrientedCuboid,
    AxisOrientedHypercuboid,
    AxisOrientedRectangle,
    EmptyRegion,
    Interval,
    LineSegment2D,
    Region,
    Region2D,
    RoiRegion,
)
from locan.dependencies import HAS_DEPENDENCY

if HAS_DEPENDENCY["open3d"]:
    pass

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    "get_region_from_intervals",
    "regions_union",
    "expand_region",
    "surrounding_region",
]


def get_region_from_intervals(
    intervals: npt.ArrayLike,
) -> (
    Interval
    | LineSegment2D
    | AxisOrientedRectangle
    | AxisOrientedCuboid
    | AxisOrientedHypercuboid
):
    """
    Constructor for instantiating an axis-oriented region
    from list of (min, max) bounds.

    Parameters
    ----------
    intervals
        The region bounds for each dimension of shape (dimension, 2).

    Returns
    -------
    Interval | AxisOrientedRectangle | AxisOrientedCuboid | AxisOrientedHypercuboid
    """
    if np.shape(intervals) == (2,) or np.shape(intervals) == (1, 2):
        return Interval.from_intervals(intervals)
    elif np.shape(intervals) == (2, 2):
        return AxisOrientedRectangle.from_intervals(intervals)
    elif np.shape(intervals) == (3, 2):
        return AxisOrientedCuboid.from_intervals(intervals)
    elif np.shape(intervals)[0] > 3 and np.shape(intervals)[1] == 2:
        return AxisOrientedHypercuboid.from_intervals(intervals)
    else:
        raise TypeError("intervals must be of shape (dimension, 2).")


def regions_union(regions: list[Region]) -> EmptyRegion | Region2D:
    """
    Return the union of `regions`.

    Parameters
    ----------
    regions
        Original region(s)

    Returns
    --------
    Region
    """
    if all([isinstance(region, (Region2D, RoiRegion)) for region in regions]):
        shapely_objects = [reg.shapely_object for reg in regions]  # type: ignore[attr-defined]
        unified_regions = unary_union(shapely_objects)
        if unified_regions.is_empty:
            return EmptyRegion()
        else:
            return Region2D.from_shapely(unified_regions)  # type: ignore
    else:
        raise NotImplementedError("regions must all be Region2D")


def expand_region(
    region: Region,
    distance: int | float = 100,
    support: Region | None = None,
    **kwargs: Any,
) -> Region:
    """
    Expand a region by `distance`.
    If region contains a list of regions, the unification of all expanded regions is
    returned.

    Parameters
    ----------
    region
        Original region(s)
    distance
        Distance by which the region is expanded orthogonal to its boundary.
    support
        A region defining the maximum outer boundary.
    kwargs
        Other parameters passed to :func:`shapely.geometry.buffer`
        for :class:`Region2D` objects.

    Returns
    --------
    Region
    """
    if distance == 0:
        expanded_region = region
    else:
        expanded_region = region.buffer(distance, **kwargs)

    if support is not None:
        expanded_region = support.intersection(expanded_region)

    try:
        return Region2D.from_shapely(expanded_region)  # type: ignore
    except AttributeError:
        return expanded_region


def surrounding_region(
    region: Region,
    distance: int | float = 100,
    support: Region | None = None,
    **kwargs: Any,
) -> Region:
    """
    Define surrounding region by extending a region and returning the extended
    region excluding the input region.
    If region contains a list of regions, the unification of all extended
    regions is returned.

    Parameters
    ----------
    region
        Original region(s)
    distance
        Distance by which the region is extended orthogonal to its boundary.
    support
        A region defining the maximum outer boundary.
    kwargs
        Other parameters passed to :func:`shapely.geometry.buffer` for
        :class:`Region2D` objects.

    Returns
    --------
    Region
    """
    extended_region = expand_region(
        region, distance=distance, support=support, **kwargs
    )
    if isinstance(extended_region, (Region2D, RoiRegion)):
        surrounding_region_ = extended_region.symmetric_difference(region)
        return Region2D.from_shapely(surrounding_region_)  # type: ignore
    else:
        raise NotImplementedError("Only 2-dimensional function has been implemented.")
