=======================
Changelog for revisions
=======================


0.2 (unreleased)
================


New Features
------------

surepy.analysis
^^^^^^^^^^^^^^^
- Implemented an analysis class Coordinate_based_colocalization.

surepy.data
^^^^^^^^^^^^^^^
- Implemented a function exclude_sparse_points to eliminate localizations in low local density regions.
- Implemented a function to apply affine coordinate transformations.
- Implemented a function to to apply a Bunwarp-transformation based on the raw transformation matrix from the ImageJ
  plugin BUnwarpJ

surepy.gui
^^^^^^^^^^^^^^^
-

surepy.io
^^^^^^^^^^^^^^^
-

surepy.render
^^^^^^^^^^^^^^^
-

surepy.scripts
^^^^^^^^^^^^^^^
-

surepy.simulation
^^^^^^^^^^^^^^^
-


API Changes
-----------

surepy.analysis
^^^^^^^^^^^^^^^
-

surepy.data
^^^^^^^^^^^^^^^
- Implemented a new region of interest management. A RoiRegion class was defined as region object in Roi objects.

surepy.gui
^^^^^^^^^^^^^^^
-

surepy.io
^^^^^^^^^^^^^^^
-

surepy.render
^^^^^^^^^^^^^^^
-

surepy.scripts
^^^^^^^^^^^^^^^
-

surepy.simulation
^^^^^^^^^^^^^^^
-

Bug Fixes
---------


surepy.analysis
^^^^^^^^^^^^^^^
-

surepy.data
^^^^^^^^^^^^^^^
- corrected index handling in track.track(), LocData.data and LocData.reduce().

surepy.gui
^^^^^^^^^^^^^^^
-

surepy.io
^^^^^^^^^^^^^^^
- changed types for column values returned from load_thunderstorm_file.

surepy.render
^^^^^^^^^^^^^^^
-

surepy.scripts
^^^^^^^^^^^^^^^
-

surepy.simulation
^^^^^^^^^^^^^^^
-


Other Changes and Additions
---------------------------

-


0.1 (9.12.2018)
========================

New Features
------------

surepy.analysis
^^^^^^^^^^^^^^^
- localization_precision
- localization_property
- localizations_per_frame
- nearest_neighbor
- pipeline
- ripley
- uncertainty

surepy.data
^^^^^^^^^^^^^^^
- cluster
- properties
- filter
- hulls
- locdata
- rois
- track
- transformation

surepy.gui
^^^^^^^^^^^^^^^
- io

surepy.io
^^^^^^^^^^^^^^^
- io_locdata

surepy.render
^^^^^^^^^^^^^^^
- render2d

surepy.scripts
^^^^^^^^^^^^^^^
- draw_roi

surepy.simulation
^^^^^^^^^^^^^^^^^^
- simulate_locdata


Other Changes and Additions
---------------------------

surepy.tests
^^^^^^^^^^^^^
- corresponding unit tests

docs
^^^^^
- rst files for sphinx documentation.

surepy
^^^^^^^
- CHANGES.rst
- LICENSE.md
- README.md
- environment.yml
- environment_dev.yml