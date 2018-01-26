.. _metadata:

========
Metadata
========

Each data class carries metadata that give details about the data history.

Each analysis class carries metadata that includes input parameters.

Metadata is saved together with data or analysis results in order to provide human- and machine-readable information
and serve for information exchange between methods.

Metadata consisting of parameter settings can e.g. be used in an analysis pipeline or for batch processing.

All metadata will be made of (nested) key:value pairs (represented by a python dictionary).

In each class metadata is a single variable with type dict.

YAML_  seems to be the preferred file format for such metadata. For a python implementation we use
ruamel.yaml_ .

.. _YAML: http://www.yaml.org/spec/1.2/spec.html
.. _ruamel.yaml: http://yaml.readthedocs.io/en/latest/index.html

This way metadata can be standardized by providing certain keywords. It is easily readable and printable. And it can
be saved in a YAML or any other text file format. It can possibly also serve as YAML header in a ASDF file format.

Structure of metadata for data classes
=======================================

LocData:
---------

    Identifier: ``str``
            Some short name.

    Comment: ``str``
            User comment.

    Production date: ``str`` (ISO standard)
            Date and time when the data was initially generated.

    Modification date: ``str`` (ISO standard)
            Date and time when the data was last modified.

    Source: ``str``
        Describes where the data is from.

        \[experiment, simulation, design, import (from some other program)]

    State: ``str``
        Indicator if data is original (as recorded) or has been modified.

        \[raw, modified]

    Experimental setup: ``dict``
        Information about the setup on which the data was generated.

    Experimental sample: ``dict``
        Information about the sample that was imaged.

    Simulation: ``dict``
        Detailed information about the simulation that was used to generate the data.

        \{'Description': '', 'Method': '', 'Parameter': {}},

    File type: ``str``
        Name of the file or program that produced the data (a fitter or simulation program).

        \[rapidStorm, Elyra, ThunderStorm, Surepy]

    File path: ``str``
        Path and name of data file.

        \[path, empty string]

    Number of elements: ``int``
        Number of elements in locdata.

    Number of frames: ``int``
        Number of frames in locdata.

    Units: ``dict``
        A dictionary with units for localization properties.

    History: ``dict``
        A list of modifications that were applied to the original data. Each element contains a dict with
        function name and parameter for the applied method.

        \[{'Method:': '', 'Parameter': {}}]

    Ancestor id: ``str`` or ``list``
        Identifier or list of identifiers of locdata from which this locdata object is derived.



Structure of metadata for analysis classes
===========================================

Analysis class:
---------------

    Analysis: ``dict``
        A dictionary with 'Method' and 'Parameter' describing the applied analysis procedure. This data can be used
        to programmatically rerun the procedure.

    Data: ``dict``
        Metadata of the input locdata on which the analysis procedure was applied.

    Comments: ``str``
        A user comment.




