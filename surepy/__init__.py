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
from surepy.constants import *
from surepy.analysis import *
from surepy.data import *
from surepy.gui import *
from surepy.io import *
from surepy.render import *
from surepy.simulation import *
from surepy.tests import *

__all__ = ['__version__']
__all__.extend(constants.__all__)
__all__.extend(analysis.__all__)
__all__.extend(data.__all__)
__all__.extend(gui.__all__)
__all__.extend(io.__all__)
__all__.extend(render.__all__)
__all__.extend(simulation.__all__)
__all__.extend(tests.__all__)
