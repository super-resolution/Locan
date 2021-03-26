"""
.. currentmodule:: locan

Locan consists of the following modules:

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
    from locan._version import version as __version__
except ImportError:
    __version__ = "0.8-not-installed"

from locan.constants import *  # constants has to be imported before any others due to pyqt5/sidepy2 issues.
from locan.analysis import *
from locan.data import *
from locan.datasets import *
from locan.gui import *
from locan.io import *
from locan.render import *
from locan.simulation import *
from locan.utils import *
from locan.tests import *


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
