.. _design:

===========================
Package Design
===========================

.. toctree::
   :hidden:

   datastructures
   metadata
   properties
   methods


Aim
==========================

We aim at providing a python package with data structures and methods for analyzing single-molecule localization data.
Locan is a Python package providing functionality to load, manipulate and analyze localization data.

The package provides:

* a class structure for localization data with appropriate meta data
* rendering functions to visualize localization data
* methods for carrying out established analysis routines
* an interface to save analysis results and run batch processes.

Locan provides standard routines and interfaces for setting up reproducible analysis pipelines and batch
processes. At the same time it allows flexibility on all programmatic levels either in Python scripts,
Jupyter notebooks or dashboard apps.
We provide original algorithms and include existing solutions by providing appropriate wrapper functions.


Outline
========

Locan provides a standardized class structure to hold and deal with :ref:`data structures <datastructures>`
such as localization data and analysis results.

:ref:`Metadata <metadata>` will be part of each data class and is either added by user input or generated during
manipulation of data classes.

:class:`LocData`, the data class for localization data, carries certain properties that describe individual or
averaged features of the underlying localizations or groups thereof. We suggest a canonical set of
:ref:`properties`.

:ref:`Methods <methods>` will either create or manipulate these data structures or perform some analysis routine
and provide the results in an appropriate form.
