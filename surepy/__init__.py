"""
.. currentmodule:: surepy

Surepy consists of the following modules:

.. autosummary::
   :toctree: generated/

   analysis
   constants
   data
   datasets
   gui
   io
   render
   scripts
   simulation
   utils
   tests
"""
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from surepy._version import version as __version__
except ImportError:
    __version__ = "0.8-not-installed"

from surepy.constants import *  # constants has to be imported before any others due to pyqt5/sidepy2 issues.
from surepy.analysis import *
from surepy.data import *
from surepy.datasets import *
from surepy.gui import *
from surepy.io import *
from surepy.render import *
from surepy.simulation import *
from surepy.utils import *
from surepy.tests import *


__all__ = ['__version__']
__all__.extend(constants.__all__)
__all__.extend(analysis.__all__)
__all__.extend(data.__all__)
__all__.extend(datasets.__all__)
__all__.extend(gui.__all__)
__all__.extend(io.__all__)
__all__.extend(render.__all__)
__all__.extend(simulation.__all__)
__all__.extend(utils.__all__)
__all__.extend(tests.__all__)
