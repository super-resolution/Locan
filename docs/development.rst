.. _development:

===========================
Development
===========================

Some things to keep in mind when adding code...

Import Conventions:
====================

The following import conventions are used throughout Surepy source code and documentation:

* import numpy as np
* import pandas as pd
* import matplotlib as mpl
* import matplotlib.pyplot as plt


Unit tests:
===========

For testing we use py.test_.

.. _py.test: https://docs.pytest.org/en/latest/index.html

A test suite is provided in Surepy/tests.

For unit testing we supply test data as data files located in tests/test_data.


To remember:
============

* The package makes use of third party packages that are only used by a few routines.

  These optional packages are treated as extra_dependencies.
  Import of optional packages is tried in the *constants* module and a `_has_package` variable is defined.
  In each module that makes use of an optional import a conditional import is carried out
  ``if _has_package: import package`` .

  In addition any user-accessible function that makes use of the optional package raises an import error
  if the optional package is not available.

* Use notation `n_something` for `number_of_something`.

* Provide commit messages with subject in imperative style (see `Chris Beams, How to Write a Git Commit Message`_).

.. _Chris Beams, How to Write a Git Commit Message: https://chris.beams.io/posts/git-commit/
