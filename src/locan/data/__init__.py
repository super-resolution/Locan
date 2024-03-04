"""
Localization data.

This module contains data models to deal with single-molecule localization data.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   hulls
   images
   locdata
   metadata_utils
   regions
   region_utils
   validation

"""

from __future__ import annotations

from importlib import import_module

import locan.data.metadata_pb2

# explicit re-exports for nested submodules are required by myp
from locan.data.hulls import (
    AlphaComplex as AlphaComplex,
    AlphaShape as AlphaShape,
    BoundingBox as BoundingBox,
    ConvexHull as ConvexHull,
    OrientedBoundingBox as OrientedBoundingBox,
)
from locan.data.locdata import *
from locan.data.images import *
from locan.data.metadata_utils import *
from locan.data.regions import (
    AxisOrientedCuboid as AxisOrientedCuboid,
    AxisOrientedHypercuboid as AxisOrientedHypercuboid,
    AxisOrientedRectangle as AxisOrientedRectangle,
    Cuboid as Cuboid,
    Ellipse as Ellipse,
    EmptyRegion as EmptyRegion,
    Interval as Interval,
    LineSegment2D as LineSegment2D,
    LineSegment3D as LineSegment3D,
    MultiPolygon as MultiPolygon,
    Polygon as Polygon,
    Rectangle as Rectangle,
    Region as Region,
    Region1D as Region1D,
    Region2D as Region2D,
    Region3D as Region3D,
    RegionND as RegionND,
    RoiRegion as RoiRegion,
    get_region_from_intervals as get_region_from_intervals,
    get_region_from_open3d as get_region_from_open3d,
    get_region_from_shapely as get_region_from_shapely,
    expand_region as expand_region,
    regions_union as regions_union,
    surrounding_region as surrounding_region,
)

from locan.data.validation import *

submodules: list[str] = [
    "hulls",
    "locdata",
    "images",
    "metadata_utils",
    "regions",
    "validation",
]

__all__: list[str] = []

for submodule in submodules:
    module_ = import_module(name=f".{submodule}", package="locan.data")
    __all__.extend(module_.__all__)
