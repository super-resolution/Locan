=======================
Changelog
=======================


0.6 (unreleased)
================


New Features
------------
-

surepy.analysis
^^^^^^^^^^^^^^^
-

surepy.data
^^^^^^^^^^^^^^^
-

surepy.datasets
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

surepy.datasets
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


0.5 (22.3.2020)
================


New Features
------------

surepy.utils
^^^^^^^^^^^^^^^
- Module surepy.utils.system_information with methods to get debugging information is added.

surepy.analysis
^^^^^^^^^^^^^^^
- LocalizationPropertyCorrelation analysis class is added.

surepy.data
^^^^^^^^^^^^^^^
- LocData.from_coordinates() is added.
- LocData.update() method is added to change dataframe with correspodning updates of hull, properties and metadata.
- Methods to compute alpha shape hulls are added.
- Pickling capability for LocData is added.

surepy.render
^^^^^^^^^^^^^^^
- scatter_2d_mpl() is added. to show locdata as scatter plot

surepy.scripts
^^^^^^^^^^^^^^^
- show_versions()


API Changes
-----------

surepy.analysis
^^^^^^^^^^^^^^^
- LocalizationProperty2D was modified and fixed.

surepy.data
^^^^^^^^^^^^^^^
- surepy.data.region_utils module is added with utility functions to analyze locdata regions.
- RoiRegions are added that support shapely Polygon and MultiPolygon objects.


Bug Fixes
---------


surepy.analysis
^^^^^^^^^^^^^^^
- Adapt colormap and rescaling in LocalizationProperty2D plot functions.


0.4.1 (16.2.2020)
=================


Bug Fixes
---------

surepy.analysis
^^^^^^^^^^^^^^^
- Fix LocalizationProperty2d fit procedure

Other Changes and Additions
---------------------------
- Increase import performance



0.4 (13.02.2020)
================

New Features
------------
- New function test() to run pytest on whole test suite.

surepy.data
^^^^^^^^^^^^^^^
- New rasterize function to divide localization support into rectangular rois.
- New functions to perform affine transformation using open3d.
- New functions to perform registration using open3d.
- New function for drift correction using icp (from open3d).
- Increase performance of maximum distance computation of localization data.

surepy.datasets
^^^^^^^^^^^^^^^
- Added functions to load example datasets. The datasets will be provided in a separate directory (repository).

surepy.scripts
^^^^^^^^^^^^^^^
- Introduced command-line interface with compound commands.
- New script to render localization data in napari
- New script to define and save rois using napari
- New script to render localizations onto raw data images


API Changes
-----------

surepy.analysis
^^^^^^^^^^^^^^^
- New analysis class for drift estimation.
- New analysis class for analysing 2d distribution of localization property.

surepy.data
^^^^^^^^^^^^^^^
- Deprecate `update_convex_hull_in_collection()`. Use `LocData.update_convex_hulls_in_references()`.
- Metadata on time is changed from timestamp to formatted time expression.

surepy.render
^^^^^^^^^^^^^^^
- Default colormaps are set to selected ones from colorcet or matplotlib.
- Add histogram function for rendering localization data.
- Add render functions to work with mpl, mpl-scatter-density, napari

surepy.scripts
^^^^^^^^^^^^^^^
- Add selection option for ellipse roi.

surepy.simulation
^^^^^^^^^^^^^^^^^^^
- Add functions for drift simulation.


Bug Fixes
---------

surepy.data
^^^^^^^^^^^^^^^
- Fixed update of bounding_box, convex_hull and oriented bounding box.


Other Changes and Additions
---------------------------
- Added centroid and dimension property to LocData.
- Implemented use of QT_API to set the QT bindings and work in combination with napari.
- Make shapely a required dependency.

0.3 (09.07.2019)
================

New Features
------------

surepy.analysis
^^^^^^^^^^^^^^^
- Added analysis class BlinkStatistics to compute on/off times in localization cluster.

surepy.data
^^^^^^^^^^^^^^^
- Introduced global variable LOCDATA_ID that serves as standard running ID for LocData objects.
- Added function update_convex_hulls_in_collection


API Changes
-----------

surepy.analysis
^^^^^^^^^^^^^^^
- Refactored all analysis class names to CamelCode.
- Refactored handling of LocData input in analysis classes to better resemble the scikit-learn API.

surepy.simulation
^^^^^^^^^^^^^^^^^^^
- Deleted deprecated simulation functions.


Other Changes and Additions
---------------------------

- Refactored all localization property names to follow the convention to start with small letters.
- Changed import organization by adding __add__ to enable import surepy as sp.
- Added dockerfiles for using and testing surepy.
- various other small changes and fixes as documented in the version control log.


0.2 (22.3.2019)
================

New Features
------------

surepy.analysis
^^^^^^^^^^^^^^^
- implemented an analysis class CoordinateBasedColocalization.
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
- sc_draw_roi_mpl

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