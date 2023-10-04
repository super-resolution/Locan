.. _installation:

===========================
Installation
===========================

Dependencies
------------

* python 3
* standard scipy and other open source libraries

A list with all hard and optional dependencies is given in `pyproject.toml`, `environment.yml` and `requirements.txt`.

Install from pypi
------------------------------

Install locan directly from the Python Package Index::

    pip install locan

Extra dependencies can be included::

    pip install locan[all]

Install from conda-forge
------------------------------

Install locan with the conda package manager::

    conda install -c conda-forge locan

Install from distribution or sources
-------------------------------------

In order to get the latest changes install from the GitHub repository
main branch::

    pip install git+https://github.com/super-resolution/Locan.git@main

or download distribution or wheel archive and install with pip::

    pip install <distribution_file>

Install from local sources::

    pip install <locan_directory>

Run tests
-----------------------

Use pytest to run the tests from the source directory::

    pytest

Or run a locan script from any directory::

    locan test


Using conda to set up a dedicated environment:
------------------------------------------------------------------------------------------

1) Install miniconda or anaconda (platform-independent)
2) Setup a new environment from the environment.yml file::

	conda env create --file "./environment.yml"

   or with specific python version::

	conda create --name locan python=3.10
	conda env update --name locan --file "./environment.yml"

3) Activate the environment and install locan.

We recommend using `mamba`_ to speed up dependency resolution::

	mamba create --name locan python=3.10
	mamba env update --name locan --file "./environment.yml"
	mamba install --name locan -c conda-forge locan
	conda activate locan

.. _mamba: https://mamba.readthedocs.io


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

You may need to install node.js for rebuilding jupyter lab::

    conda install -c conda-forge nodejs

Various installation issues
-----------------------------

1) Numba requires specific numpy version:

    Numba might not be compatible with the latest numpy version.

    Solution: Install numba first.


2) Building a wheel for hdbscan raises error:

    Building a wheel for hdbscan during installation might cause the following error:

    "ValueError: numpy.ndarray size changed, may indicate binary incompatibility. "

    This error arises from version incompatibility between the numpy version installed in the current environment
    and the one used for building the wheel.

    Solution: Install wheel, cython, and numpy (or numba) with oldest-supported-numpy first, then build hdbscan using the installed versions
    (and not in isolation as done by default), and finally install locan::

        pip install wheel cython numba oldest-supported-numpy
        pip install hdbscan --no-build-isolation
        pip install locan[all]

3) Running "locan napari" in conda environment raises error:

    Starting napari in a conda environment with python>=3.8 and pyside2 causes the following error:

    "RuntimeError: PySide2 rcc binary not found in..."

    Seems like napari>0.4.5 does not work with pyside2<5.14 due to the replacement of
    pyside2-uic/pyside2-rcc.

    Solution: Set up an environment with python 3.7.
    Or set up an environment with only pyqt5 instead of pyside2.
    Or, if both pyqt5 and pyside2 are installed, set the environment variable "QT_API"::

        import os
        os.environ["QT_API"] = "pyqt5"

