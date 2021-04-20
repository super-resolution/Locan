.. _methods:

=========================
Methods for data analysis
=========================

Locan will provide methods to work on :class:`LocData` objects and carry out standard analysis procedures. Some of these
functions are merely wrapper functions for well established analysis routines in third-party packages.

The methods in this context can either be stand-alone functions with a well-defined input and output and absolutely
no side-effects.

In general however, results consist of numeric data, statistical properties (e.g. a histogram of numeric results)
and/or fit parameter from comparing the numeric results with theoretical expectations, and annotated plots. In this case
the method will be part of a more complex analysis class.

Locan therefore provides specific analysis classes that follow a common structure.
Any :class:`Analysis` object is instantiated with a set of parameters that define the precise analysis procedure.
The computation is then performed by calling :meth:`Analysis.compute` with a parameter specifying one or more
:class:`LocData` objects on which to perform the analysis.
The main result is typically provided under the attribute :attr:`Analysis.results` and accompanied by a flexible set of
further attributes. Common methods often include plot, hist, and report functions. Metadata is provided under the
:attr:`Analysis.meta` attribute.
