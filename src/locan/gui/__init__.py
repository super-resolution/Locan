"""

User interfaces.

This module provides functions and classes for using graphical user interfaces (GUI).

Functions provide a GUI based on QT if the QT backend and appropriate python bindings
are available. Supported bindings are found in the enum class `locan.QtBindings`.

The configuration variable `locan.QT_BINDING` declares which python binding to use
if several are installed.
If an environment variable `QT_API` is defined it takes precedence over
`locan.QT_BINDING`.

If neither `locan.QT_BINDING` nor `QT_API` is defined, qtpy will choose the binding.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   io

"""
from __future__ import annotations

from locan.gui import io

from .io import *

__all__: list[str] = []
__all__.extend(io.__all__)
