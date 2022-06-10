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
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
import re
import sys
from collections.abc import Iterable
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

__all__ = [
    "needs_package",
    "IMPORT_NAMES",
    "INSTALL_REQUIRES",
    "EXTRAS_REQUIRE",
    "HAS_DEPENDENCY",
    "QtBindings",
]


def _has_dependency_factory(
    packages: Iterable[str], import_names: dict[str, str] | None = None
) -> dict[str, bool]:
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
    import_names : dict[str, str] | None
        Mapping of import names on package names.
    has_dependency : dict | None
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
                raise ImportError(
                    f"Function {func} needs {import_name} which cannot be imported."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _get_dependencies(package: str) -> tuple[set[str], set[str]]:
    """Read out all required and optional (extra) dependencies for locan (PyPI names)."""
    requires = importlib.metadata.requires(package)

    pattern = r"[\w-]+"
    required_dependencies = {
        re.match(pattern, item).group() for item in requires if "extra ==" not in item
    }
    extra_dependencies = {
        re.match(pattern, item).group() for item in requires if "extra ==" in item
    }

    return required_dependencies, extra_dependencies


#: List of required dependencies (PyPi package names)
INSTALL_REQUIRES = set()

#: List of optional dependencies (PyPi package names)
EXTRAS_REQUIRE = set()
INSTALL_REQUIRES, EXTRAS_REQUIRE = _get_dependencies(package="locan")

#: A dictionary mapping PyPi package names to import names if they are different
IMPORT_NAMES = dict()
IMPORT_NAMES["ruamel"] = "ruamel.yaml"
IMPORT_NAMES["fast-histogram"] = "fast_histogram"
IMPORT_NAMES["boost-histogram"] = "boost_histogram"
IMPORT_NAMES["protobuf"] = "google.protobuf"
IMPORT_NAMES["scikit-image"] = "skimage"
IMPORT_NAMES["scikit-learn"] = "sklearn"
IMPORT_NAMES["pytest-qt"] = "pytestqt"
IMPORT_NAMES["mpl-scatter-density"] = "mpl_scatter_density"

#: A dictionary indicating if dependency is available.
HAS_DEPENDENCY = _has_dependency_factory(
    packages=INSTALL_REQUIRES.union(EXTRAS_REQUIRE)
)


# Possible python bindings to interact with QT
class QtBindings(Enum):
    """
    Python bindings to interact with Qt.
    """

    NONE = ""
    PYSIDE2 = "pyside2"
    PYQT5 = "pyqt5"
    PYSIDE6 = "pyside6"
    PYQT6 = "pyqt6"


def _set_qt_binding(qt_binding: QtBindings | str) -> str:
    """
    Check if qtpy can import `qt_binding` and return the qt_binding that will be used.
    Checks os.environ["QT_API"] first.
    If os.environ["QT_API"] is set it will take precedence over `qt_bindings`.

    Note
    -----
    This function must be used before qtpy is imported for the first time.
    """
    QT_API = ""
    if "qtpy" in sys.modules:
        logger.warning("qtpy has already been loaded. QT_BINDING cannot be changed.")
    elif os.environ.get("QT_API", ""):
        QT_API = os.environ["QT_API"]
    else:
        if isinstance(qt_binding, QtBindings):
            QT_API = qt_binding.value
        elif qt_binding:
            QT_API = qt_binding.lower()

        if QT_API:
            os.environ["QT_API"] = QT_API

    from qtpy import API

    if QT_API and QT_API != API:
        logger.warning(f"QT_BINDING {QT_API} is not available - {API} is used instead.")

    return API
