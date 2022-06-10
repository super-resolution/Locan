.. _gui:

==========================
User Interface
==========================

Locan is mostly designed to work without a graphical user interface (gui).
However, a few methods make use of a gui or interact with third-party libraries that
provide a gui.

Locan and various other libraries make use of `qt` as a gui backend if it is installed
together with appropriate python bindings. Different python bindings exist to interact
with `qt` including `pyside2` and `pyqt5`.

Locan makes use of `pyqt` to choose a binding.
To force locan and other libraries to use a specific binding you can set the
python environment variable `QT_API` before importing
any QT-dependent package:

.. code:: python

   import os
   os.environ["QT_API"] = "pyside2"

For more details see also the documentation on the :mod:`locan.gui` module.