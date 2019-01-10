'''
Compute additional properties for locdata objects.

These functions take locdata as input, compute one or more new properties, and return a dict with the property names
as key and the corresponding values.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   cbc
   max_distance
   statistics
'''

from surepy.data.properties.max_distance import *
from surepy.data.properties.statistics import *
from surepy.data.properties.cbc import *
