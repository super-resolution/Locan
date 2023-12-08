"""

Region of interest.

This module provides functions for managing regions of interest in
localization data.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   roi
"""
from __future__ import annotations

from locan.rois import roi

from .roi import *

__all__: list[str] = []
__all__.extend(roi.__all__)
