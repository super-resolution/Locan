"""
Process localization data.

This module contains procedures to process localization data to yield new
datasets.
All functions typically take LocData objects as input and return LocData
properties or whole new LocData objects.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   aggregate
   cluster
   filter
   register
   properties
   tracking
   transform
"""

from __future__ import annotations

from importlib import import_module

# explicit re-exports for nested submodules are required by myp
from locan.process.aggregate import *
from locan.process.cluster import (
    cluster_by_bin as cluster_by_bin,
    cluster_dbscan as cluster_dbscan,
    cluster_hdbscan as cluster_hdbscan,
    serial_clustering as serial_clustering,
)
from locan.process.filter import *
from locan.process.register import *
from locan.process.properties import (
    distance_to_region as distance_to_region,
    distance_to_region_boundary as distance_to_region_boundary,
    inertia_moments as inertia_moments,
    max_distance as max_distance,
    range_from_collection as range_from_collection,
    ranges as ranges,
    statistics as statistics,
)
from locan.process.tracking import *
from locan.process.transform import (
    bunwarp as bunwarp,
    overlay as overlay,
    standardize as standardize,
    transform_affine as transform_affine,
    transform_counts_to_photons as transform_counts_to_photons,
)

submodules: list[str] = [
    "aggregate",
    "cluster",
    "filter",
    "register",
    "properties",
    "tracking",
    "transform",
]

__all__: list[str] = []

for submodule in submodules:
    module_ = import_module(name=f".{submodule}", package="locan.process")
    __all__.extend(module_.__all__)
