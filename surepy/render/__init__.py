"""

Render localization data.

This module provides functions to render and present localization data.
It makes use of the matplotlib environment.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   render2d

"""

from .render2d import *
from .utilities import *

__all__ = []
__all__.extend(render2d.__all__)
__all__.extend(utilities.__all__)
