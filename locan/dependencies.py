"""
Module to deal with required and optional dependencies.

Optional dependencies are defined in setup.cfg as [options.extras_require].

In any module that requires an optional dependency the import should be carried out on top:
    if HAS_DEPENDENCY["package"]: import package

Any function that makes use of the optional dependency should be decorated with
    @needs_package("package")


CONSTANTS

.. autosummary::
   :toctree: ./

   INSTALL_REQUIRES
   EXTRAS_REQUIRE
   IMPORT_NAMES
   HAS_DEPENDENCY
   QT_BINDINGS
"""
import os
from functools import wraps
import importlib.util
import configparser
import re
from enum import Enum
import warnings
from typing import Optional, Union, Dict

from locan import ROOT_DIR


__all__ = ["needs_package", "IMPORT_NAMES", "INSTALL_REQUIRES", "EXTRAS_REQUIRE", "HAS_DEPENDENCY",
           "QtBindings", "QT_BINDINGS"]


def _has_dependency_factory(
        packages: list,
        import_names: Optional[Dict[str, bool]] = None
) -> dict:
    if import_names is None:
        import_names = IMPORT_NAMES
    has_dependency = dict()
    for package in packages:
        key = import_names.get(package, package)
        value = importlib.util.find_spec(key) is not None
        has_dependency[key] = value
    return has_dependency


def needs_package(package, import_names=None, has_dependency=None):
    """
    Function that returns a decorator to check for optional dependency.

    Parameters
    ----------
    package : str
        Package or dependency name that needs to be imported.
    import_names : Optional[Dict, None]
        Mapping of import names on package names.
    has_dependency : Optional[Dict, None]
        Dictionary with bool indicator if package (import name) is available.

    Returns
    -------
    callable
        A decorator that raises ImportError if package is not available.
    """
    if import_names is None:
        import_names = IMPORT_NAMES
    if has_dependency is None:
        has_dependency = HAS_DEPENDENCY
    import_name = import_names.get(package, package)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not has_dependency[import_name]:
                raise ImportError(f"Function {func} needs {import_name} which cannot be imported.")
            return func(*args, **kwargs)
        return wrapper

    return decorator


#: List of required dependencies (PyPi package names)
# Should reflect the dependencies specified in setup.cfg.
# Some package names are different from the names for import.
INSTALL_REQUIRES = ['asdf', 'tifffile', 'ruamel.yaml', 'fast-histogram', 'boost-histogram', 'hdbscan', 'lmfit',
                    'protobuf', 'shapely', 'networkx', 'scikit-learn', 'scikit-image', 'matplotlib', 'scipy',
                    'pandas', 'numpy', 'tqdm', 'numba', 'cython']

#: List of optional dependencies (PyPi package names)
EXTRAS_REQUIRE = {
    "pytest",
    "colorcet",
    "trackpy",
    "open3d",
    "PySide2", "napari", "mpl_scatter_density",
    "requests", "h5py",
    "cupy",
    "sphinx", "ipython", "myst-nb", "sphinx-copybutton", "sphinx_rtd_theme", "furo",
    "coverage", "build", "twine"
}

#: A dictionary mapping PyPi package names to import names if they are different
IMPORT_NAMES = dict()
IMPORT_NAMES["ruamel"] = "ruamel.yaml"
IMPORT_NAMES["fast-histogram"] = "fast_histogram"
IMPORT_NAMES["boost-histogram"] = "boost_histogram"
IMPORT_NAMES["protobuf"] = "google.protobuf"
IMPORT_NAMES["scikit-image"] = "skimage"
IMPORT_NAMES["scikit-learn"] = "sklearn"

#: A dictionary indicating if dependency is available.
HAS_DEPENDENCY = _has_dependency_factory(packages=EXTRAS_REQUIRE)


# Possible python bindings to interact with QT
class QtBindings(Enum):
    """
    Python bindings used to interact with Qt.
    """
    NONE = 'none'
    PYSIDE2 = 'pyside2'
    PYQT5 = 'pyqt5'


# Force python binding - only for testing purposes:
# os.environ['QT_API'] = QtBindings.PYQT5.value


# Determine Python bindings for QT interaction.
if 'QT_API' in os.environ:  # this is the case that QT_API has been set.
    try:
        QT_BINDINGS = QtBindings(os.environ['QT_API'])
    except KeyError:
        warnings.warn(f'The requested QT_API {os.environ["QT_API"]} cannot be imported. Will continue without QT.')
        QT_BINDINGS = QtBindings.NONE

    if QT_BINDINGS == QtBindings.PYSIDE2:
        if importlib.util.find_spec("PySide2") is None:
            warnings.warn(f'The requested QT_API {QT_BINDINGS.value} cannot be imported. Will continue without QT.')
            QT_BINDINGS = QtBindings.NONE

    elif QT_BINDINGS == QtBindings.PYQT5:
        if importlib.util.find_spec("PyQt5") is None:
            warnings.warn(f'The requested QT_API {QT_BINDINGS.value} cannot be imported. Will continue without QT.')
            QT_BINDINGS = QtBindings.NONE

else:  # this is the case that QT_API has not been set.
    if importlib.util.find_spec("PySide2") is not None:
        QT_BINDINGS = QtBindings.PYSIDE2
    elif importlib.util.find_spec("PyQt5") is not None:
        QT_BINDINGS = QtBindings.PYQT5
    else:
        QT_BINDINGS = QtBindings.NONE  # this is the case that no qt bindings are available.

# In order to force napari and other QT-using libraries to import with the correct Qt bindings
# the environment variable QT_API has to be set.
# See use of qtpy in napari which default to pyqt5 if both bindings are installed.
if QT_BINDINGS != QtBindings.NONE:
    os.environ['QT_API'] = QT_BINDINGS.value
