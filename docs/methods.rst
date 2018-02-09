.. _methods:

========
Methods
========

Surepy will provide methods to work on the basic data structures and carry out analysis procedures.

The methods in this context can either be stand-alone functions with a well-defined input and output and absolutely
no side-effects;
or they can be part of a class with additional attributes, e.g. carrying complex analysis results, and class methods,
e.g. providing plots as matplotlib axes object.

Methods for data manipulation:
--------------------------------

1) Methods that return or manipulate Dataset objects:
    * load_rapidSTORM_file(path) -> Dataset
    * load_Elyra_file(path) -> Dataset
    * simulate() -> Dataset

    * reduce(selection) -> Dataset

2) Methods that return or manipulate Selection objects:
    * select_by_condition(dataset, criterion) -> Selection
    * select_by_region(dataset, region) -> Selection
    * select_by_image_mask(dataset, image) -> Selection

    * exclude_sparce_points(selection, threshold_density) -> Selection
    * transform(selection, threshold_density) -> Selection
    * drift_correction(selection) -> Selection

    * reduce(collection) -> Selection

3) Methods that return or manipulate Collection objects:
    * clustering(selection, kwargs) -> Collection
    * select_by_condition(collection, condition) -> Collection
    * load(path) -> Collection


Methods for data analysis:
---------------------------

Analysis methods take one or more data classes as input, and then generate and provide results.

If the result consists of numeric data the method should be a simple function.

In general however, results consist if numeric data, statistical properties (e.g. a histogram of numeric results)
or fit parameter from comparing the numeric results with theoretical expectations, and annotated plots. In this case
the method will be part of a more complex class:

    class Analysis()
        Input parameter:
            * Selection or Collection
        Attributes:
            * numeric results
            * statistics of the results (e.g. histogram)
            * fit parameter from fitting a model function to numeric results or statistics
            * plot

Methods can be classified depending on their input:

1) Methods that take a selection as input:
*******************************************************

    * Localizations_per_frame()
        Compute the number of localizations in each frame.
    * Localization_precision()
	    Estimating localization precision from spatial differences between successive localizations.

3) Methods that take two selections as input:
*******************************************************

    * ...

4) Methods that take a collection of selections as input:
**********************************************************

    * ...

5) Methods that take two or more collections as input:
*******************************************************

    * ...

Methods for rendering data:
---------------------------

Rendering methods take one or more selections as input and render an image.