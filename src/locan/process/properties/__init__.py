"""
Compute additional properties for locdata objects.

These functions take locdata as input, and compute one or more new properties.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   misc
   locdata_statistics
"""

from __future__ import annotations

from locan.process.properties.locdata_statistics import *
from locan.process.properties.misc import *

from . import locdata_statistics, misc

__all__: list[str] = []
__all__.extend(locdata_statistics.__all__)
__all__.extend(misc.__all__)
