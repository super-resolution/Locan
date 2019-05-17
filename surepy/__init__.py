"""
.. currentmodule:: surepy

Surepy consists of the following modules:

.. autosummary::
   :toctree: generated/

   analysis
   data
   gui
   io
   render
   scripts
   simulation
   tests
   constants

"""

from surepy.version import __version__
from surepy.data.locdata import LocData
from surepy.analysis import *
from surepy.data import *

__all__ = ['__version__', 'LocData']
__all__.extend(analysis.__all__)
__all__.extend(data.__all__)
