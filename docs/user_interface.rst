.. _gui:

==========================
User Interface
==========================

Surepy is mostly designed to work without a general user interface (gui).
However, a few methods make use of a gui or interact with third party libraries that provide a gui.

Various libraries make use of `qt` as a gui backend if it is installed together with appropriate python bindings.
Different python bindings exist to interact with `qt` including `pyqt5` and `pyside2`.

Surepy by itself defaults to using `pyside2`.

However, `matplotlib`, `napari` and `surepy` take different defaults if both `pyqt5` and `pyside2` are installed in
the same python environment. If different bindings are used at import time a RunTime Error will arise.

To force all libraries to use the same binding you must set the python environment variable `QT_API` before importing
any package:

.. code:: python

   import os
   os.environ["QT_API"] = "pyside2"

For more details see also the documentation on the `surepy.gui` module.