"""

Render localization data with matplotlib.

This module provides functions to render and present localization data
making use of matplotlib.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   render2d
   render3d

"""
from __future__ import annotations

from locan.visualize.render_mpl import render2d, render3d

from .render2d import *
from .render3d import *

__all__: list[str] = []
__all__.extend(render2d.__all__)  # type: ignore
__all__.extend(render3d.__all__)  # type: ignore
