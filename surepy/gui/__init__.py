"""

User interfaces.

This module provides functions and classes for using graphical user interfaces (GUI).
Functions provide a GUI based on QT if the QT backend and appropriate python bindings are available.

The constant `surepy.constants.QT_BINDINGS` declares which python binding to use.
Depending on the situation it is set to one of the values specified in the enum class `surepy.constants.QtBindings`.

1) `QtBindings.NONE` if no QT backend with python bindings is installed.

2) `QtBindings.PYSIDE2` or `QtBindings.PYQT5` if exactly one of them is installed.

3) the `QtBindings` value that is equal to the python environment variable `QT_API`
   if both PySide2 and PyQt5 are installed and `QT_API` exists.

4) `QtBindings.PYSIDE2` if both PySide2 and PyQt5 are installed and `QT_API` does not exist.

To force surepy to use specific QT bindings set the `QT_API` environment variable to 'pyside2' or 'pyqt5'.


Submodules:
-----------

.. autosummary::
   :toctree: ./

   io

"""
from .io import *

__all__ = []
__all__.extend(io.__all__)