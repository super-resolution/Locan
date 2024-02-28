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

   spatial_transform
   intensity_transform
   temporal_transform
   filter
   register
   aggregate
   filter
   transform
   cluster
   tracking
   properties
"""

from __future__ import annotations

from importlib import import_module

from locan.process.aggregate import *
from locan.process.cluster import *
from locan.process.filter import *
from locan.process.register import *
from locan.process.properties import *
from locan.process.tracking import *
from locan.process.transform import *

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
