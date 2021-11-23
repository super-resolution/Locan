.. _installation:

===========================
Installation
===========================

Dependencies
------------

* python 3
* standard scipy and other open source libraries

A list with all requirements is given in `setup.cfg`, `environment.yml` and `requirements.txt`.

Install from pypi
------------------------------

    pip install locan

Install from distribution
------------------------------

Download distribution or wheel archive and install with pip::

    pip install <distribution_file>

Install from source directory
------------------------------

Install from sources without extra requirements::

    pip install <locan_directory>

Install from sources with extra requirements::

    pip install <locan_directory>[all]

Run tests
-----------------------

Use pytest to run the tests from the source directory::

    pytest

Or run a locan script from any directory::

    locan test


Using Conda to set up a dedicated environment:
------------------------------------------------------------------------------------------

1) Install miniconda or anaconda (platform-independent)
2) Setup a new environment from the environment.yml file::

	conda env create --file "./environment.yml"

3) Install locan using pip

Jupyter
-----------------------

To work with jupyter notebooks install jupyter lab::

    pip install jupyterlab

or inside a conda environment::

    conda install -c conda-forge jupyterlab

Make sure to add the appropriate lab extensions::

    jupyter labextension install @jupyter-widgets/jupyterlab-manager \
                                 @pyviz/jupyterlab_pyviz \
                                 jupyter-matplotlib

You may need to install node.js for rebuilding jupyter-lab::

    conda install -c conda-forge nodejs
