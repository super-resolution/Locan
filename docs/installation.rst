.. _installation:

===========================
Installation
===========================

Dependencies
------------

* python 3
* standard scipy and other open source libraries

A list with all requirements is given in `environment.yml`.


Install from source directory
------------------------------

Use setuptools to install Surepy from sources::

    python setup.py install


Run tests
-----------------------

Use setuptools (which is configured for running pytest) to run the tests::

    python setup.py test


Install from pypi
------------------------------

Do not use `pip install surepy` as there is an independent package called surepy on pypi.


Using Conda to set up a dedicated environment:
------------------------------------------------------------------------------------------

1) Install miniconda or anaconda (platform-independent)
2) Setup a new environment from the environment.yml file::

	conda env create --file "./environment.yml"

3) Activate the new directory::

    conda activate surepy

4) Install surepy from sources::

    python setup.py install

3) If needed install additional packages using the environment files for tutorials or development::

    conda env update --file "./environment_dev.yml"
