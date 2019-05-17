'''
Compute additional properties for locdata objects.

These functions take locdata as input, compute one or more new properties, and return a dict with the property names
as key and the corresponding values.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   max_distance
   statistics
'''

from surepy.data.properties.distances import *
from surepy.data.properties.locdata_statistics import *

__all__ = []
__all__.extend(distances.__all__)
__all__.extend(locdata_statistics.__all__)