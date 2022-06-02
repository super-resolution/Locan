.. _installation:

===========================
Installation
===========================

Dependencies
------------

* python 3
* standard scipy and other open source libraries

A list with all hard and optional dependencies is given in `setup.cfg`, `environment.yml` and `requirements.txt`.

Install from pypi
------------------------------

Install locan directly from the Python Package Index::

    pip install locan

Extra dependencies can be included::

    pip install locan[all]

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

   or with specific python version::

	conda create --name locan python==3.9
	conda env update --name locan --file "./environment.yml"

3) Activate the environment and install locan using pip

We recommend using mamba to speed up dependency resolution.

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
