=======================
Changelog
=======================

0.10 (unreleased)
=================


New Features
------------
-

locan.analysis
^^^^^^^^^^^^^^^
-

locan.data
^^^^^^^^^^^^^^^
-

locan.datasets
^^^^^^^^^^^^^^^
-

locan.gui
^^^^^^^^^^^^^^^
-

locan.io
^^^^^^^^^^^^^^^
-

locan.render
^^^^^^^^^^^^^^^
-

locan.scripts
^^^^^^^^^^^^^^^
-

locan.simulation
^^^^^^^^^^^^^^^^^
-


API Changes
-----------

locan.analysis
^^^^^^^^^^^^^^^
-

locan.data
^^^^^^^^^^^^^^^
-

locan.datasets
^^^^^^^^^^^^^^^
-

locan.gui
^^^^^^^^^^^^^^^
-

locan.io
^^^^^^^^^^^^^^^
-

locan.render
^^^^^^^^^^^^^^^
-

locan.scripts
^^^^^^^^^^^^^^^
-

locan.simulation
^^^^^^^^^^^^^^^^^^^
-

Bug Fixes
---------

locan.analysis
^^^^^^^^^^^^^^^
-

locan.data
^^^^^^^^^^^^^^^
-

locan.gui
^^^^^^^^^^^^^^^
-

locan.io
^^^^^^^^^^^^^^^
-

locan.render
^^^^^^^^^^^^^^^
-

locan.scripts
^^^^^^^^^^^^^^^
-

locan.simulation
^^^^^^^^^^^^^^^^^
-


Other Changes and Additions
---------------------------
-

0.9 - 2021-06-12
================

API Changes
-----------

locan.data
^^^^^^^^^^^^^^^
- Restructured Region management introducing new classes in locan.data.region

locan.simulation
^^^^^^^^^^^^^^^^^^^
- Refactored simulation functions to make use of numpy random number generator.
- Refactored simulation functions to generate Neyman-Scott point processes in expanded regions.

Other Changes and Additions
---------------------------
- Added ro modified tutorials on mutiprocessing, regions and simulation.


0.8 - 2021-05-06
================

API Changes
-----------

locan.scripts
^^^^^^^^^^^^^^^
- Default values for verbose and extra flags in script show_versions were changed.

Bug Fixes
---------

locan.analysis
^^^^^^^^^^^^^^^
- Fit procedure was fixed for NearestNeighborDistances.

Other Changes and Additions
---------------------------
- Library was renamed to LOCAN
- Documentation and tutorials were modified accordingly
- Test coverage was improved and use of coverage.py introduced
- _future module was deprecated


0.7 - 2021-03-26
================

API Changes
------------

locan.analysis
^^^^^^^^^^^^^^^
- Added new keyword parameters in LocData.from_chunks and Drift.
- Extended class for blinking analysis.

Other Changes and Additions
---------------------------
- Turn warning into log for file io.
- Restructured documentation, added tutorials, and changed html-scheme to furo.


0.6 - 2021-03-04
================

New Features
------------
- Introduced logging capability.
- Added script for running tests from command line interface.

locan.analysis
^^^^^^^^^^^^^^^
- Make all analysis classes pickleable.
- Refactored Pipeline class
- Enabled and tested multiprocessing based on multiprocessing or ray.
- Added more processing bars.
- Added drift analysis and correction based on imagecorrelation and iterative closest point registration.

locan.data
^^^^^^^^^^^^^^^
- Made LocData class pickleable.
- Added computation of inertia moments.
- Added orientation property based on oriented bounding box and inertia moments.
- Added elongation property based on oriented bounding box.
- Add transformation method to overlay LocData objects.

locan.io
^^^^^^^^^^^^^^^
- Added loading function for Nanoimager data.

locan.render
^^^^^^^^^^^^^^^
- Added windowing function for image data.

API Changes
-----------

locan.analysis
^^^^^^^^^^^^^^^
- Fixed and extended methods for Drift analysis and correction.

locan.data
^^^^^^^^^^^^^^^
- Implemented copy and deepcopy for LocData.
- Changed noise output in clustering methods. Removed noise parameter.

locan.datasets
^^^^^^^^^^^^^^^
- Added dataset for microtubules

locan.io
^^^^^^^^^^^^^^^
- Added option for file-like objects in io_locdata functions.
- Added Bins class, introduced use of boost-histogram package, and restructured binning.
- Introduced use of napari.run.
- Changed default value in render_2d_mpl to interpolation='nearest'.

locan.scripts
^^^^^^^^^^^^^^^
- Added arguments for locan napari and locan rois.

locan.simulation
^^^^^^^^^^^^^^^^^^^
- Added simulation of frame values.

Bug Fixes
---------

locan.data
^^^^^^^^^^^^^^^
- Fixed treatment of empty LocData in clustering and hull functions.

locan.gui
^^^^^^^^^^^^^^^
- Use PySide2 as default QT backend depending on QT_API setting.

locan.io
^^^^^^^^^^^^^^^
- Fixed enconding issues for loading Elyra data.

Other Changes and Additions
---------------------------
- Test data is included in distribution.
- New dockerfiles for test and deployment.
- Included pyproject.toml file


0.5.1 - 2020-03-25
==================
- Update environment and requirement files


0.5 - 2020-03-22
================


New Features
------------

locan.utils
^^^^^^^^^^^^^^^
- Module locan.utils.system_information with methods to get debugging information is added.

locan.analysis
^^^^^^^^^^^^^^^
- LocalizationPropertyCorrelation analysis class is added.

locan.data
^^^^^^^^^^^^^^^
- LocData.from_coordinates() is added.
- LocData.update() method is added to change dataframe with correspodning updates of hull, properties and metadata.
- Methods to compute alpha shape hulls are added.
- Pickling capability for LocData is added.

locan.render
^^^^^^^^^^^^^^^
- scatter_2d_mpl() is added. to show locdata as scatter plot

locan.scripts
^^^^^^^^^^^^^^^
- show_versions()


API Changes
-----------

locan.analysis
^^^^^^^^^^^^^^^
- LocalizationProperty2D was modified and fixed.

locan.data
^^^^^^^^^^^^^^^
- locan.data.region_utils module is added with utility functions to analyze locdata regions.
- RoiRegions are added that support shapely Polygon and MultiPolygon objects.


Bug Fixes
---------


locan.analysis
^^^^^^^^^^^^^^^
- Adapt colormap and rescaling in LocalizationProperty2D plot functions.


0.4.1 - 2020-02-16
==================


Bug Fixes
---------

locan.analysis
^^^^^^^^^^^^^^^
- Fix LocalizationProperty2d fit procedure

Other Changes and Additions
---------------------------
- Increase import performance



0.4 - 2020-02-13
================

New Features
------------
- New function test() to run pytest on whole test suite.

locan.data
^^^^^^^^^^^^^^^
- New rasterize function to divide localization support into rectangular rois.
- New functions to perform affine transformation using open3d.
- New functions to perform registration using open3d.
- New function for drift correction using icp (from open3d).
- Increase performance of maximum distance computation of localization data.

locan.datasets
^^^^^^^^^^^^^^^
- Added functions to load example datasets. The datasets will be provided in a separate directory (repository).

locan.scripts
^^^^^^^^^^^^^^^
- Introduced command-line interface with compound commands.
- New script to render localization data in napari
- New script to define and save rois using napari
- New script to render localizations onto raw data images


API Changes
-----------

locan.analysis
^^^^^^^^^^^^^^^
- New analysis class for drift estimation.
- New analysis class for analysing 2d distribution of localization property.

locan.data
^^^^^^^^^^^^^^^
- Deprecate `update_convex_hull_in_collection()`. Use `LocData.update_convex_hulls_in_references()`.
- Metadata on time is changed from timestamp to formatted time expression.

locan.render
^^^^^^^^^^^^^^^
- Default colormaps are set to selected ones from colorcet or matplotlib.
- Add histogram function for rendering localization data.
- Add render functions to work with mpl, mpl-scatter-density, napari

locan.scripts
^^^^^^^^^^^^^^^
- Add selection option for ellipse roi.

locan.simulation
^^^^^^^^^^^^^^^^^^^
- Add functions for drift simulation.


Bug Fixes
---------

locan.data
^^^^^^^^^^^^^^^
- Fixed update of bounding_box, convex_hull and oriented bounding box.


Other Changes and Additions
---------------------------
- Added centroid and dimension property to LocData.
- Implemented use of QT_API to set the QT bindings and work in combination with napari.
- Make shapely a required dependency.

0.3 - 2019-07-09
================

New Features
------------

locan.analysis
^^^^^^^^^^^^^^^
- Added analysis class BlinkStatistics to compute on/off times in localization cluster.

locan.data
^^^^^^^^^^^^^^^
- Introduced global variable LOCDATA_ID that serves as standard running ID for LocData objects.
- Added function update_convex_hulls_in_collection


API Changes
-----------

locan.analysis
^^^^^^^^^^^^^^^
- Refactored all analysis class names to CamelCode.
- Refactored handling of LocData input in analysis classes to better resemble the scikit-learn API.

locan.simulation
^^^^^^^^^^^^^^^^^^^
- Deleted deprecated simulation functions.


Other Changes and Additions
---------------------------

- Refactored all localization property names to follow the convention to start with small letters.
- Changed import organization by adding __add__ to enable import locan as sp.
- Added dockerfiles for using and testing locan.
- various other small changes and fixes as documented in the version control log.


0.2 - 2019-03-22
================

New Features
------------

locan.analysis
^^^^^^^^^^^^^^^
- implemented an analysis class CoordinateBasedColocalization.
- implemented an analysis class AccumulationClusterCheck.

locan.data
^^^^^^^^^^^^^^^
- implemented a function exclude_sparse_points to eliminate localizations in low local density regions.
- implemented a function to apply affine coordinate transformations.
- implemented a function to to apply a Bunwarp-transformation based on the raw transformation matrix from the ImageJ
  plugin BUnwarpJ

locan.simulation
^^^^^^^^^^^^^^^^^
- implemented functions to simulate localization data based on complete spatial randomness, Thomas, or Matern processes.
- implemented functions simulate_xxx to provided LocData objects.
- implemented functions make_xxx to provide point coordinates.


API Changes
-----------

locan.data
^^^^^^^^^^^^^^^
- implemented a new region of interest management. A RoiRegion class was defined as region object in Roi objects.


Bug Fixes
---------

locan.data
^^^^^^^^^^^^^^^
- corrected index handling in track.track(), LocData.data and LocData.reduce().

locan.io
^^^^^^^^^^^^^^^
- changed types for column values returned from load_thunderstorm_file.


0.1 - 2018-12-09
========================

New Features
------------

locan.analysis
^^^^^^^^^^^^^^^
- localization_precision
- localization_property
- localizations_per_frame
- nearest_neighbor
- pipeline
- ripley
- uncertainty

locan.data
^^^^^^^^^^^^^^^
- cluster
- properties
- filter
- hulls
- locdata
- rois
- track
- transformation

locan.gui
^^^^^^^^^^^^^^^
- io

locan.io
^^^^^^^^^^^^^^^
- io_locdata

locan.render
^^^^^^^^^^^^^^^
- render2d

locan.scripts
^^^^^^^^^^^^^^^
- sc_draw_roi_mpl

locan.simulation
^^^^^^^^^^^^^^^^^^
- simulate_locdata


Other Changes and Additions
---------------------------

locan.tests
^^^^^^^^^^^^^
- corresponding unit tests

docs
^^^^^
- rst files for sphinx documentation.

locan
^^^^^^^
- CHANGES.rst
- LICENSE.md
- README.md
- environment.yml
- environment_dev.yml