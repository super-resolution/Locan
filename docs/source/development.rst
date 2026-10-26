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

        pip install --group dev --group docs .[all,pyside6]

Alternatively, in a conda environment you may use the `environment_dev.yml`.

With uv_ the complete development environment (pinned to the Python version in
`.python-version`, with all optional dependencies and one Qt binding) is::

        uv sync --extra all --extra pyside6

.. _uv: https://docs.astral.sh/uv/

Import Conventions
====================

The following import conventions are used throughout Locan source code and
documentation:

* import boost_histogram as bh
* import matplotlib as mpl
* import matplotlib.pyplot as plt
* import networkx as nx
* import numpy as np
* import open3d as o3d
* import pandas as pd
* import scipy as sp

This is enforced through ruff following specifications in pyproject.toml.

Unit tests
===========

For testing we use pytest_.

.. _pytest: https://docs.pytest.org/en/latest/index.html

A minimal test suite is provided in `src/locan/tests`.
The extended test suite is provided in `tests`.

For unit testing we supply test data as data files located in `tests/test_data`.

Configurations are kept in pyproject.toml.

Coverage
===========

For measuring code coverage in testing we use coverage.py_.

.. _coverage.py: https://coverage.readthedocs.io

Configurations are kept in pyproject.toml.

Code checks
============

We use black_ for formating and ruff_ for code linting.

.. _black: https://pypi.org/project/black/
.. _ruff: https://pypi.org/project/ruff

Configurations are kept in pyproject.toml.

We use pyrefly_ for type checking.

.. _pyrefly: https://pyrefly.org/

Configurations are kept in pyproject.toml.

Versioning
===========

We use `SemVer`_ for versioning. For all versions available, see the
`releases in this repository`_.

.. _SemVer: http://semver.org/
.. _releases in this repository: https://github.com/super-resolution/Locan/releases

Coding agents
==============

Repository-specific conventions for AI coding agents are summarized in
`AGENTS.md` at the repository root; it is kept consistent with this page.

To remember:
============

* The package makes use of third party packages that are only used by a few routines.

  These optional packages are declared as extras in `pyproject.toml`.
  Their availability is determined once in the :mod:`locan.dependencies`
  module and recorded in the mapping ``HAS_DEPENDENCY``.
  In each module that makes use of an optional import a conditional import is
  carried out

  .. code:: python

     from locan.dependencies import HAS_DEPENDENCY, needs_package

     if HAS_DEPENDENCY["package"]:
         import package

  In addition any user-accessible function that makes use of the optional
  package is decorated with ``@needs_package("package")`` and thereby raises an
  import error if the optional package is not available.

  Test functions that require optional dependencies should be marked with:

  .. code:: python

   @pytest.mark.skipif(not HAS_DEPENDENCY["package"], reason="requires package")


* The length of code and docstring lines should be kept to 88 and
  75 characters, respectively.

* Use notation `n_something` for `number_of_something`.


* Provide commit messages with subject in imperative style
  (see `Chris Beams, How to Write a Git Commit Message`_).
  If possible follow the specifications for `conventional commits`_.

.. _Chris Beams, How to Write a Git Commit Message: https://chris.beams.io/posts/git-commit/
.. _conventional commits: https://www.conventionalcommits.org
