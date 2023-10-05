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
   rois
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

from locan.data.aggregate import *
from locan.data.cluster import *
from locan.data.filter import *
from locan.data.hulls import *
from locan.data.locdata import *
from locan.data.metadata_utils import *
from locan.data.properties import *
from locan.data.region import *
from locan.data.region_utils import *
from locan.data.register import *
from locan.data.rois import *
from locan.data.tracking import *
from locan.data.transform import *
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
    "rois",
    "tracking",
    "transform",
    "validation",
]

__all__: list[str] = []

for submodule in submodules:
    module_ = import_module(name=f".{submodule}", package="locan.data")
    __all__.extend(module_.__all__)
