.. _methods:

========
Methods
========

Surepy will provide methods to work on LocData objects and carry out standard analysis procedures. Some of these
functions are merely wrapper functions for well established analysis routines in third-party packages.

The methods in this context can either be stand-alone functions with a well-defined input and output and absolutely
no side-effects;
or they can be methods of a class providing additional class attributes, e.g. complex analysis results,
and class methods, e.g. plots in the form of matplotlib axes objects.

Methods for data analysis:
---------------------------

Analysis methods take one or more data classes as input, and then generate and provide results.

If the result consists of numeric data the method should be a simple function.

In general however, results consist if numeric data, statistical properties (e.g. a histogram of numeric results)
or fit parameter from comparing the numeric results with theoretical expectations, and annotated plots. In this case
the method will be part of a more complex analysis class.

    class Analysis()
        Input parameter:
            * LocData objects
        Attributes:
            * numeric results
            * statistics of the results (e.g. histogram)
            * fit parameter from fitting a model function to numeric results or statistics
            * plot
