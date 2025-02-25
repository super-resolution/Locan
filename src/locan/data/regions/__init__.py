"""
Region objects for localization data.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   region
   region_utils

"""

from __future__ import annotations

from locan.data.regions.region import *
from locan.data.regions.region_utils import *

from . import region, region_utils

__all__: list[str] = []
__all__.extend(region.__all__)
__all__.extend(region_utils.__all__)
