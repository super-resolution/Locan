.. _design:

===========================
Package Design
===========================

Outline:
========

We aim at designing a standard class structure to hold and deal with localization data and analysis results
(see :ref:`Data structures <datastructures>`).

:ref:`Metadata <metadata>` will be part of each data class and either added by user input or generated during
manipulation of data classes.

Locdata, the data class for localization data, carries certain properties that describe individual or
averaged features of the underlying localizations or groups thereof. We suggest a canonical set of
:ref:`properties`.

:ref:`Methods <methods>` will either create or manipulate these data structures or perform some analysis routine
and provide the results in an appropriate form.

The project is organized by this :ref:`directory structure <directories>`.
