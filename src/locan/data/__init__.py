"""
Localization data.

This module contains data models to deal with single-molecule localization data.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   locdata
   hulls
   region
   region_utils
   metadata_utils
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
from locan.data.metadata_utils import *
from locan.data.regions import *
from locan.data.validation import *

submodules: list[str] = [
    "hulls",
    "locdata",
    "metadata_utils",
    "regions",
    "validation",
]

__all__: list[str] = []

for submodule in submodules:
    module_ = import_module(name=f".{submodule}", package="locan.data")
    __all__.extend(module_.__all__)
