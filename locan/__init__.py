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
   scripts
   simulation
   utils
   visualize
   tests
"""
from __future__ import annotations

import logging
from pathlib import Path

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from locan._version import version as __version__
except ImportError:
    __version__ = "not-installed"


# Root directory for path operations.
ROOT_DIR: Path = Path(__file__).parent

# Identifier for LocData objects that is reset for each locan session and incremented with each LocData instantiation.
locdata_id: int = 0


# isort: off
from locan.dependencies import *  # imported by locan.configuration
from locan.constants import *  # imported by locan.configuration
from locan.configuration import *  # has to be imported before any others due to pyqt5/sidepy2 issues.
from locan.analysis import *
from locan.data import *
from locan.datasets import *
from locan.gui import *
from locan.locan_io import *
from locan.simulation import *
from locan.utils import *
from locan.visualize import *
from locan.tests import *

# isort: on

__all__: list[str] = ["__version__", "ROOT_DIR"]
__all__.extend(dependencies.__all__)  # type: ignore
__all__.extend(constants.__all__)  # type: ignore
__all__.extend(configuration.__all__)  # type: ignore
__all__.extend(analysis.__all__)  # type: ignore
__all__.extend(data.__all__)  # type: ignore
__all__.extend(datasets.__all__)  # type: ignore
__all__.extend(gui.__all__)  # type: ignore
__all__.extend(locan_io.__all__)  # type: ignore
__all__.extend(simulation.__all__)  # type: ignore
__all__.extend(utils.__all__)  # type: ignore
__all__.extend(visualize.__all__)  # type: ignore
__all__.extend(tests.__all__)  # type: ignore
