"""

Render localization data.

This module provides functions to render and present localization data.
It mostly makes use of the matplotlib environment.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   render2d
   render3d
   utilities
   transform

"""

from .render2d import *
from .render3d import *
from .utilities import *
from .transform import *

__all__ = []
__all__.extend(render2d.__all__)
__all__.extend(render3d.__all__)
__all__.extend(utilities.__all__)
__all__.extend(transform.__all__)
