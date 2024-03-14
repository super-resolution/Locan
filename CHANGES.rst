=======================
Changelog
=======================

0.19.1 - 2024-03-14
=================

Other Changes and Additions
---------------------------
- constrain protobuf version to <5

0.19 - 2023-12-12
=================

Other Changes and Additions
---------------------------
- add lazy_imports of API
- introduce src-layout in code base

0.18 - 2023-12-06
=================

API Changes
-----------
- add colormaps module and change management of colormaps
- remove outdated module with analysis example

Bug Fixes
---------
- fix version readout with readthedocs

Other Changes and Additions
---------------------------
- add GitHub action for deploying to PyPI and TestPyPI
- configure setuptools_scm for branching model

0.17 - 2023-10-26
=================

New Features
------------
- feat: adapt LocData.from_dataframe to take any dataframe that supports the dataframe interchange protocol.
- feat: add gui dialog to set file path

API Changes
-----------
- remove deprecated LocalizationUncertaintyFromIntensity

Bug Fixes
---------
- fix: tests for shapely 2.0.2 with geos 3.12.0 in conda environment
- fix: bin-to-pixel relation in render in napari
- fix: readthedocs version readout

Other Changes and Additions
---------------------------
- refactor: update to python 3.12

0.16 - 2023-10-05
=================

API Changes
-----------
- refactor: delete qt and needs_datasets marker for pytest

Bug Fixes
---------
- fix: import of pytest for type checking

Other Changes and Additions
---------------------------
- refactor: reduce load in GitHub actions
- add locan to conda-forge

0.15 - 2023-09-28
=================

New Features
------------
- feat: Selector class to specify loc_property selections

API Changes
-----------
- refactor: change return type of clustering algorithms for noise from None to LocData()

Bug Fixes
---------
- fix: convert_property_types and add documentation.

Other Changes and Additions
---------------------------
- Add type hints
- Add type checking with mypy in pre-commit and GitHub actions CI workflow
- Extend ruff linting
- refactor(test): include pytest-qt in standard tests.
- drop support for python 3.8


0.14 - 2023-06-30
=================

New Features
------------
- Add analysis routine for `SubpixelBias`
- Add function `merge_metadata` to merge metadata from class instance or file
- Add class `Files` for managing file selections to be used in batch processing
- Add `utilities/statistics` module with helper function to compute
  `WeightedMeanVariance`
- Add `locdata.utilities.statistics.ratio_fwhm_to_sigma` function
- Add `locdata.utilities.statistics.biased_variance` function
- Add analysis routine for `ConvexHullExpectation`,
  `GroupedPropertyExpectation` and `PositionVarianceExpectation`.
- Add function to standardize locdata.
- add `Locdata.update_properties_in_references` method
- add analysis class `locdata.analysis.ConvexHullExpectationBatch`.

API Changes
-----------
- Change `find_pattern_upstream` into `find_file_upstream`
- Add analysis class `LocalizationUncertainty` and
  deprecate `LocalizationUncertaintyFromIntensity`
- Refactor `LocData.properties` to include weighted coordinate averages
  and properties for frame and intensity
- Refactor `locan.simulation.simulate_locdata.resample`
- Change `LoData.coordinate_properties` to `LocData.coordinate_keys` and
  add `LocData.uncertainty_keys` and corresponding functions in
  `locan.constants.PropertyKeys`

Bug Fixes
---------
- Fix use of callables in `LocalizationUncertainty`
- Adapt to to bug fixes in lmfit 1.2.0

Other Changes and Additions
---------------------------
- Use ruff for linting
- Use pyproject.toml for all specifications and deprecate use of setup.cfg
- Use import open3d-cpu instead of open3d
- docs: add tutorial for analysis of grouped cluster properties

0.13 - 2023-02-15
=================

New Features
------------
- Add CLI for --info and --version
- Add overlay function to transform locdatas
- Add function to standardize locdata.
- Add function load_metadata_from_toml to read metadata from toml file
- Add function find_pattern_upstream.
- Add function transform_counts_to_photons.

API Changes
-----------
- Add boost_histogram_axes property to Bins

Bug Fixes
---------
- Fix intensity transform with nan
- Fix simulation on region being a rotated rectangle
- Fix Roi.from_yaml for rotated rectangular rois
- Fix Roi.from_yaml to work with Polygon regions
- Fix bug in localization_property_2d.
- Fix bug in localizations_per_frame concerning the time units
- Fix histogram with bins=Bins_instance as input
- Fix conversion of bin_edges to bin_size for float numbers
- Fix tests by excluding shapely 2.0.0 and 2.0.1
- Use np.random.default_rng for random number generation in simulate_drift.py

Other Changes and Additions
---------------------------
- Use of pre-commit
- Adapt to isort, black, flake8, bandit
- Integrate qtpy
- Add benchmark setup for Airspeed Velocity
- Modify dockerfiles to run with slim-bullseye
- Add conditional import of tomllib to replace tomli for python>=3.11
- Add CITATION.cff file
- Add some type hints


0.12 - 2022-06-02
=================

API Changes
-----------

locan.configuration
^^^^^^^^^^^^^^^^^^^^^
- Introduced module locan.configuration to hold user-specific configuration values

locan.constants
^^^^^^^^^^^^^^^^^
- Introduced enum `PropertyKey` that holds `PropertyDescription` dataclasses
  with information on name, type, units and description

locan.data
^^^^^^^^^^^^^^^
- Provided new scheme for metadata
- Added tutorial about metadata for LocData
- Introduced use of protobuf Timestamp and Duration types in metadata messages
- Added function in `locan.data.metadata_utils` to provide scheme of default metadata
- Added function in `locan.data.metadata_utils` to read metadata from toml file

locan.io
^^^^^^^^^^^^^^^
- Refactored locan_io.locdata module structure

locan.render
^^^^^^^^^^^^^^^
- Changed default value for n_bins in HistogramEqualization to increase intensity resolution.
  Note: This modification changes the visual presentation of localization data with a large dynamic range.

Bug Fixes
---------

locan.utils.system_information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed `show_version` to read out all dependency versions

locan.data
^^^^^^^^^^^^^^^
- Fixed bug in cluster functions such that setting the region of empty collections does not raise a TypeError anymore
- Fixed protobuf issues related to protobuf 4.21

Other Changes and Additions
---------------------------
- Dropped support for python 3.7
- Various minor changes of documentation and code
- Removed numba as dependency
- Based conda-related dockerfiles on mambaforge
- Introduced use of fixture from pytest-qt for testing QT interfaces

0.11.1 - 2022-04-08
===================

Bug Fixes
---------

locan.scripts
^^^^^^^^^^^^^^^
- Fix a bug introduced in 0.11 in napari and rois script.


0.11 - 2022-03-22
=================

New Features
------------

locan.data
^^^^^^^^^^^^^^^
- Modified Polygon.contains function to increase performance.
- Implemented randomize function for all hull types.

locan.io
^^^^^^^^^^^^^^^
- Added methods to load DECODE and SMAP files.

locan.render
^^^^^^^^^^^^^^^
- Added rendering functions for 3D
- Added rendering functions for RGB image (multi-color overlay)

API Changes
-----------

locan.io
^^^^^^^^^^^^^^^
- Extended load_txt_files to convert property names to locan standard property names.

locan.render
^^^^^^^^^^^^^^^
- Refactored intensity rescaling by introducing standard normalization procedures.

Bug Fixes
---------

locan.data
^^^^^^^^^^^^^^^
- Fixed bunwarp transformation

locan.io
^^^^^^^^^^^^^^^
- Fixed lineterminator in load_rapidstorm_track_file

Other Changes and Additions
---------------------------
- Ensured support of locan and all optional dependencies for Python 3.9
- Ensured support of locan (without optional dependencies) for Python 3.10
- Turned hdbscan into optional dependency

0.10 - 2021-11-23
=================

New Features
------------

locan.io
^^^^^^^^^^^^^^^
- Add function to load rapidSTORM file with tracked data.
- Add function to load and save SMLM file.

Other Changes and Additions
---------------------------
- Locan went public.
- Readthedocs was set up.
- Zenodo DOI was generated.


0.9 - 2021-11-11
================

API Changes
-----------

locan.analysis
^^^^^^^^^^^^^^^
- Refactor computation of blinking_statistics

locan.data
^^^^^^^^^^^^^^^
- Restructured Region management introducing new classes in locan.data.region
- Rename function to computer inertia moments

locan.render
^^^^^^^^^^^^^^^
- Change image orientation in render_2d_napari to be consistent with points coordinates.

locan.simulation
^^^^^^^^^^^^^^^^^^^
- Refactored simulation functions to make use of numpy random number generator.
- Refactored simulation functions to generate Neyman-Scott point processes in expanded regions.
- Add function to simulate dSTORM data as localization clusters
  with normal-distributed coordinates and geometric-distributed number of localizations.

Other Changes and Additions
---------------------------
- Added or modified tutorials on mutiprocessing, regions and simulation.
- Introduce pytest markers for test functions that are excluded from test run per default.

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
- Introduced global variable locdata_id that serves as standard running ID for LocData objects.
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