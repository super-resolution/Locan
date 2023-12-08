"""
Localization data.

This module contains classes and functions to deal with single-molecule localization data.
All functions provide or modify LocData objects.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   locdata
   properties
   hulls
   region
   region_utils
   register
   aggregate
   filter
   transform
   cluster
   tracking
   metadata_utils
   validation

"""
from __future__ import annotations

from importlib import import_module

import locan.data.metadata_pb2

# explicit re-exports for nested submodules are required by myp
from locan.data.aggregate import *
from locan.data.cluster import (
    cluster_by_bin as cluster_by_bin,
    cluster_dbscan as cluster_dbscan,
    cluster_hdbscan as cluster_hdbscan,
    serial_clustering as serial_clustering,
)
from locan.data.filter import *
from locan.data.hulls import (
    AlphaComplex as AlphaComplex,
    AlphaShape as AlphaShape,
    BoundingBox as BoundingBox,
    ConvexHull as ConvexHull,
    OrientedBoundingBox as OrientedBoundingBox,
)
from locan.data.locdata import *
from locan.data.metadata_utils import *
from locan.data.properties import (
    distance_to_region as distance_to_region,
    distance_to_region_boundary as distance_to_region_boundary,
    inertia_moments as inertia_moments,
    max_distance as max_distance,
    range_from_collection as range_from_collection,
    ranges as ranges,
    statistics as statistics,
)
from locan.data.region import *
from locan.data.region_utils import *
from locan.data.register import *
from locan.data.tracking import *
from locan.data.transform import (
    bunwarp as bunwarp,
    overlay as overlay,
    standardize as standardize,
    transform_affine as transform_affine,
    transform_counts_to_photons as transform_counts_to_photons,
)
from locan.data.validation import *

submodules: list[str] = [
    "aggregate",
    "cluster",
    "filter",
    "hulls",
    "locdata",
    "metadata_utils",
    "properties",
    "region",
    "region_utils",
    "register",
    "tracking",
    "transform",
    "validation",
]

__all__: list[str] = []

for submodule in submodules:
    module_ = import_module(name=f".{submodule}", package="locan.data")
    __all__.extend(module_.__all__)
