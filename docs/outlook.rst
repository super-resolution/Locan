.. _outlook:

===========================
Outlook
===========================

Surepy will provide data structures and methods for working with single-molecule localization data. Here we outline
long term development ideas.


Logging:
=================

For user-controlled batch processing and analysis pipelines a logging procedure will help to control and supervise the
process. Logging could be introduced based on the logging package in the standard python library.


Analysis Methods:
=================

Surepy provides methods to work on locdata and perform localization data
analysis. On the long run several analysis routines will be helpful:


    * Fourier_shell_correlation()
        Compute fourier shell correlation for localization coordinates as measure of resolution.

    * Statistics_report()
        Provide a printable form of selected statistics on locdata properties.

    * Quality_report()
        Provide a printable form of selected analysis procedures to check imaging quality.

    * Region_overlay()
        Compute spatial overlay of hulls from two dataset with locdata hulls.
