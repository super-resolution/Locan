=======================
Changelog for revisions
=======================


0.3 (unreleased)
================


New Features
------------

surepy.analysis
^^^^^^^^^^^^^^^
-

surepy.data
^^^^^^^^^^^^^^^
-

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
^^^^^^^^^^^^^^^^^
-


API Changes
-----------

surepy.analysis
^^^^^^^^^^^^^^^
-

surepy.data
^^^^^^^^^^^^^^^
-

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
^^^^^^^^^^^^^^^^^^^
-

Bug Fixes
---------


surepy.analysis
^^^^^^^^^^^^^^^
-

surepy.data
^^^^^^^^^^^^^^^
-

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
^^^^^^^^^^^^^^^^^
-


Other Changes and Additions
---------------------------

-


0.2 (22.3.2019)
================

New Features
------------

surepy.analysis
^^^^^^^^^^^^^^^
- implemented an analysis class Coordinate_based_colocalization.
- implemented an analysis class AccumulationClusterCheck.

surepy.data
^^^^^^^^^^^^^^^
- implemented a function exclude_sparse_points to eliminate localizations in low local density regions.
- implemented a function to apply affine coordinate transformations.
- implemented a function to to apply a Bunwarp-transformation based on the raw transformation matrix from the ImageJ
  plugin BUnwarpJ

surepy.simulation
^^^^^^^^^^^^^^^^^
- implemented functions to simulate localization data based on complete spatial randomness, Thomas, or Matern processes.
- implemented functions simulate_xxx to provided LocData objects.
- implemented functions make_xxx to provide point coordinates.


API Changes
-----------

surepy.data
^^^^^^^^^^^^^^^
- implemented a new region of interest management. A RoiRegion class was defined as region object in Roi objects.


Bug Fixes
---------

surepy.data
^^^^^^^^^^^^^^^
- corrected index handling in track.track(), LocData.data and LocData.reduce().

surepy.io
^^^^^^^^^^^^^^^
- changed types for column values returned from load_thunderstorm_file.


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