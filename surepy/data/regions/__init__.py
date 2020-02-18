"""
Define and work with geometric regions.


Submodules:
-----------

.. autosummary::
   :toctree: ./

   region
   utils

"""

from surepy.data.regions.region import *
from surepy.data.regions.utils import *

__all__ = []
__all__.extend(region.__all__)
__all__.extend(utils.__all__)
