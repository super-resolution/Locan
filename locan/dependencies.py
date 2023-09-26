"""
Module to deal with required and optional dependencies.

Optional dependencies are defined in pyproject.toml.

In any module that requires an optional dependency the import should be
conditioned:
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
from collections.abc import Callable, Iterable
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "needs_package",
    "IMPORT_NAMES",
    "INSTALL_REQUIRES",
    "EXTRAS_REQUIRE",
    "HAS_DEPENDENCY",
    "QtBindings",
]

F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")
T = TypeVar("T")


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


def needs_package(
    package: str,
    import_names: dict[str, str] | None = None,
    has_dependency: dict[str, bool] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Function that returns a decorator to check for optional dependency.

    Parameters
    ----------
    package
        Package or dependency name that needs to be imported.
    import_names
        Mapping of package names onto import names.
    has_dependency
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

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not has_dependency[import_name]:  # type: ignore
                raise ImportError(
                    f"Function {func} needs {import_name} which cannot be imported."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _get_dependencies(package: str) -> tuple[set[str], set[str]]:
    """
    Read out all required and optional (extra) dependencies for locan.

    Returns
    -------
    tuple[set[str], set[str]]
        PyPI names
    """
    requires = importlib.metadata.requires(package)
    if requires is None:
        required_dependencies: set[str] = set()
        extra_dependencies: set[str] = set()
    else:
        pattern = r"[\w-]+"
        required_dependencies = set()
        extra_dependencies = set()
        for item in requires:
            match = re.match(pattern, item)
            if match is None:
                pass
            else:
                if "extra ==" in item:
                    extra_dependencies.add(match.group())
                else:
                    required_dependencies.add(match.group())
    return required_dependencies, extra_dependencies


#: List of required dependencies (PyPi package names)
INSTALL_REQUIRES: set[str] = set()

#: List of optional dependencies (PyPi package names)
EXTRAS_REQUIRE: set[str] = set()
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
    This function must be used before qtpy is imported for the first time
    to be effective.

    Returns
    -------
    str
        qt-bindings or empty string
    """
    if isinstance(qt_binding, QtBindings):
        qt_api = qt_binding.value
    elif qt_binding:
        qt_api = qt_binding.lower()
    else:
        qt_api = ""

    if "qtpy" in sys.modules:
        logger.warning("qtpy has already been loaded. QT_BINDING cannot be changed.")
    elif os.environ.get("QT_API", ""):
        qt_api = os.environ["QT_API"]
    else:
        if qt_api:
            os.environ["QT_API"] = qt_api

    try:
        from qtpy import (  # noqa: F401  # import API alone does not raise ModuleNotFoundError if no module is available
            API,
            QtCore,
        )

        if qt_api and qt_api != API:
            logger.warning(
                f"QT_BINDING {qt_api} is not available - {API} is used instead."
            )
            os.environ["QT_API"] = API
        qt_api = API
    except ImportError:
        if qt_api:
            logger.warning("QT_BINDING is not available.")
        qt_api = ""

    return qt_api
