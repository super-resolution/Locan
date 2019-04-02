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

* Provide commit messages with subject in imperative style (see `Chris Beams, How to Write a Git Commit Message`_).

.. _Chris Beams, How to Write a Git Commit Message: https://chris.beams.io/posts/git-commit/
