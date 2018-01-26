.. _design:

===========================
Package Design
===========================

Outline:
========

We aim at designing a standard class structure to hold and deal with localization data and analysis results
(see :ref:`Data structures <datastructures>`).

:ref:`Metadata <Metadata>` will be part of each data class and either added by user input or generated during
manipulation of data classes.

:ref:`Methods <Methods>` will either create or manipulate these data structures or perform some analysis routine
and provide the results in an appropriate form.

An overview on useful methods that will be designed in the long run is given in :ref:`outlook`.

.. toctree::
   :maxdepth: 1
   :caption: Design Considerations

   datastructures
   properties
   metadata
   methods
   outlook
   directories
   documentation
   development
