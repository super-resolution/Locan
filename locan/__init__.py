"""
.. currentmodule:: locan

Locan consists of the following modules:

.. autosummary::
   :toctree: generated/

   analysis
   constants
   configuration
   data
   datasets
   dependencies
   gui
   locan_io
   render
   scripts
   simulation
   utils
   tests
"""
from pathlib import Path
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from locan._version import version as __version__
except ImportError:
    __version__ = "0.12-not-installed"


# Root directory for path operations.
ROOT_DIR = Path(__file__).parent

# Identifier for LocData objects that is reset for each locan session and incremented with each LocData instantiation.
locdata_id = 0


from locan.dependencies import *  # has to be imported before any others due to pyqt5/sidepy2 issues.
from locan.constants import *
from locan.configuration import *
from locan.analysis import *
from locan.data import *
from locan.datasets import *
from locan.gui import *
from locan.locan_io import *
from locan.render import *
from locan.simulation import *
from locan.utils import *
from locan.tests import *


__all__ = ['__version__', 'ROOT_DIR']
__all__.extend(dependencies.__all__)
__all__.extend(constants.__all__)
__all__.extend(configuration.__all__)
__all__.extend(analysis.__all__)
__all__.extend(data.__all__)
__all__.extend(datasets.__all__)
__all__.extend(gui.__all__)
__all__.extend(locan_io.__all__)
__all__.extend(render.__all__)
__all__.extend(simulation.__all__)
__all__.extend(utils.__all__)
__all__.extend(tests.__all__)
