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


__all__ = ['__version__', 'LocData']
