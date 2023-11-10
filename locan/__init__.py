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
   tests
   utils
   visualize
"""
from __future__ import annotations

import lazy_loader as lazy
import logging
from pathlib import Path

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from locan._version import version as __version__
except ImportError:
    __version__ = "not-installed"


__getattr__, __lazy_dir__, __lazy_all__ = lazy.attach_stub(
    package_name=__name__, filename=__file__
)

__all__ = __lazy_all__ + ["__version__", "ROOT_DIR", "locdata_id"]


# Root directory for path operations.
ROOT_DIR: Path = Path(__file__).parent

# Identifier for LocData objects that is reset for each locan session and incremented with each LocData instantiation.
locdata_id: int = 0


def __dir__():
    return __lazy_dir__() + ["__version__", "ROOT_DIR", "locdata_id"]
