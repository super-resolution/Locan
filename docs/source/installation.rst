.. _installation:

===========================
Installation
===========================

Dependencies
------------

* python 3
* standard scipy and other open source libraries

A list with all requirements is given in `setup.cfg`, `environment.yml` and `requirements.txt`.

Install from distribution
------------------------------

Download distribution or wheel archive and install with pip::

    pip install <distribution_file>

Install from source directory
------------------------------

Use setuptools to install from sources (without extra requirements)::

    python setup.py install

Alternatively, use pip to install from sources with extra requirements::

    pip install .[all]

Run tests
-----------------------

Use setuptools (which is configured for running pytest) to run the tests::

    python setup.py test

Or run a locan script from any directory::

    locan test


Using Conda to set up a dedicated environment:
------------------------------------------------------------------------------------------

1) Install miniconda or anaconda (platform-independent)
2) Setup a new environment from the environment.yml file::

	conda env create --file "./environment.yml"


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
