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

    * Property_correlation()
        Present and analyze the correlation between two locdata properties.

    * Statistics_report()
        Provide a printable form of selected statistics on locdata properties.

    * Quality_report()
        Provide a printable form of selected analysis procedures to check imaging quality.

    * Coordinate_based_colocalization()
        Compute localization-based colocalization values for two distinct sets of locdata coordinates.

    * Region_overlay()
        Compute spatial overlay of hulls from two dataset with locdata hulls.

    * Point_region_distances()
        Analyze distances between locdata coordinates and hull regions in a locdata collection (center and boundary).
