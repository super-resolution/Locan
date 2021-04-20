.. _development:

===========================
Development
===========================

We welcome any contributions for improving or further developing this package.

However, please excuse that we are limited in time for development and support.

Some things to keep in mind when adding code...

Install
========

A few extra libraries are needed for development::

        pip install .[dev]

Alternatively, you may use the requirement files `requirements_dev.txt` or `environment_dev.yml`.


Import Conventions
====================

The following import conventions are used throughout Locan source code and documentation:

* import numpy as np
* import pandas as pd
* import matplotlib as mpl
* import matplotlib.pyplot as plt
* import open3d as o3d
* import networkx as nx
* import boost_histogram as bh


Unit tests
===========

For testing we use py.test_.

.. _py.test: https://docs.pytest.org/en/latest/index.html

A test suite is provided in locan/tests.

For unit testing we supply test data as data files located in locan/tests/test_data.

Versioning
===========

We use [SemVer](http://semver.org/) for versioning. For all versions available, see the
[releases in this repository](https://github.com/super-resolution/Locan/releases).


To remember:
============

* The package makes use of third party packages that are only used by a few routines.

  These optional packages are treated as extra_dependencies.
  Import of optional packages is tried in the *constants* module and a `_has_package` variable is defined.
  In each module that makes use of an optional import a conditional import is carried out

  .. code:: python

     if _has_package: import package

  In addition any user-accessible function that makes use of the optional package raises an import error
  if the optional package is not available.

  Test functions that require optional dependencies should be marked with:

  .. code:: python

   @pytest.mark.skipif(not _has_package,reason="requires optional package")


* Use notation `n_something` for `number_of_something`.


* Provide commit messages with subject in imperative style (see `Chris Beams, How to Write a Git Commit Message`_).

.. _Chris Beams, How to Write a Git Commit Message: https://chris.beams.io/posts/git-commit/
