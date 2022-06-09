"""

Synthetic data

This module provides functions to simulate localization and other data that can be used for testing and development.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   simulate_locdata
   simulate_drift

"""
from .simulate_locdata import *
from .simulate_drift import *


__all__ = []
__all__.extend(simulate_locdata.__all__)
__all__.extend(simulate_drift.__all__)
