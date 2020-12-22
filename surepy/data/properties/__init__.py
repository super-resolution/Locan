"""
Compute additional properties for locdata objects.

These functions take locdata as input, and compute one or more new properties.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   misc
   statistics
"""

from surepy.data.properties.misc import *
from surepy.data.properties.locdata_statistics import *

__all__ = []
__all__.extend(misc.__all__)
__all__.extend(locdata_statistics.__all__)