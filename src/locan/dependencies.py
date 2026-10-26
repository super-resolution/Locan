"""
Module to deal with required and optional dependencies.

Optional dependencies are defined in pyproject.toml.

In any module that requires an optional dependency the import should be
conditioned:

    if HAS_DEPENDENCY["package"]: import package

Any function that makes use of the optional dependency should be decorated with

    @needs_package("package")

Keys of :data:`HAS_DEPENDENCY` are import names. Distribution names from
pyproject.toml are normalized (PEP 503) and mapped through :data:`IMPORT_NAMES`;
a distribution without an entry is looked up under its name with ``-``
replaced by ``_``. Looking up a name that is not a dependency of locan yields
``False`` (with a warning logged once) instead of raising ``KeyError``.


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
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from functools import wraps
from typing import Any, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "needs_package_version",
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


def _normalize_name(name: str) -> str:
    """
    Normalize a distribution name as defined in PEP 503.

    Parameters
    ----------
    name
        Distribution (PyPI) name in any spelling.

    Returns
    -------
    str
        Lower-case name with runs of ``-``, ``_`` and ``.`` collapsed to ``-``.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _import_name(package: str, import_names: Mapping[str, str] | None = None) -> str:
    """
    Map a distribution name onto the name under which it is imported.

    Parameters
    ----------
    package
        Distribution (PyPI) name in any spelling.
    import_names
        Mapping of normalized distribution names onto import names.
        Defaults to :data:`IMPORT_NAMES`.

    Returns
    -------
    str
        Import name: the mapped name if there is an entry, else the package
        name with ``-`` replaced by ``_``.
    """
    if import_names is None:
        import_names = IMPORT_NAMES
    normalized = _normalize_name(package)
    if normalized in import_names:
        return import_names[normalized]
    if package in import_names:
        return import_names[package]
    return package.replace("-", "_")


def _is_importable(import_name: str) -> bool:
    """
    Check if a module can be found without importing it.

    ``find_spec`` of a dotted name imports the parent package, which raises
    if that parent is missing; this is reported as "not available".
    """
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


class _HasDependency(dict[str, bool]):
    """
    Mapping of import names onto availability.

    Names that are no dependency of locan (e.g. when locan is not installed as
    a distribution and its metadata cannot be read) are reported as not
    available instead of raising ``KeyError``; a warning is logged once per
    name.
    """

    def __init__(self, mapping: Mapping[str, bool] | None = None) -> None:
        super().__init__(mapping if mapping is not None else {})
        self._warned: set[str] = set()

    def __missing__(self, key: str) -> bool:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(
                f"{key} is not a known dependency of locan "
                "and is treated as not available."
            )
        return False


def _has_dependency_factory(
    packages: Iterable[str], import_names: Mapping[str, str] | None = None
) -> dict[str, bool]:
    has_dependency = _HasDependency()
    for package in packages:
        key = _import_name(package, import_names=import_names)
        has_dependency[key] = _is_importable(key)
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
    if has_dependency is None:
        has_dependency = HAS_DEPENDENCY
    import_name = _import_name(package, import_names=import_names)

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
    Read out all required and optional (extra) dependencies for package.

    Parameters
    ----------
    package
        Package or dependency name to be checked.

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

#: A dictionary mapping normalized PyPi package names (PEP 503) to import names
#: if they differ in more than ``-`` versus ``_``.
IMPORT_NAMES: dict[str, str] = dict()
IMPORT_NAMES["ruamel"] = "ruamel.yaml"
IMPORT_NAMES["protobuf"] = "google.protobuf"
IMPORT_NAMES["scikit-image"] = "skimage"
IMPORT_NAMES["scikit-learn"] = "sklearn"
IMPORT_NAMES["pytest-qt"] = "pytestqt"
# the next three are implied by the "-" to "_" fallback; kept for compatibility
IMPORT_NAMES["fast-histogram"] = "fast_histogram"
IMPORT_NAMES["boost-histogram"] = "boost_histogram"
IMPORT_NAMES["mpl-scatter-density"] = "mpl_scatter_density"
IMPORT_NAMES["cython"] = "Cython"
IMPORT_NAMES["open3d-cpu"] = "open3d"
IMPORT_NAMES["pyside2"] = "PySide2"
IMPORT_NAMES["pyside6"] = "PySide6"
IMPORT_NAMES["pyqt5"] = "PyQt5"
IMPORT_NAMES["pyqt6"] = "PyQt6"

#: A dictionary indicating if dependency (import name) is available.
#: ``HAS_DEPENDENCY["qt"]`` tells if qtpy can load a Qt binding; it is refined by
#: :func:`_set_qt_binding` once the configured binding is known.
HAS_DEPENDENCY: dict[str, bool] = _has_dependency_factory(
    packages=INSTALL_REQUIRES.union(EXTRAS_REQUIRE).union(set(["pytestqt"]))
)


# Possible python bindings to interact with QT
class QtBindings(StrEnum):
    """
    Python bindings to interact with Qt.
    """

    NONE = ""
    PYSIDE2 = "pyside2"
    PYQT5 = "pyqt5"
    PYSIDE6 = "pyside6"
    PYQT6 = "pyqt6"


HAS_DEPENDENCY["qt"] = _is_importable("qtpy") and any(
    _is_importable(_import_name(binding.value))
    for binding in QtBindings
    if binding is not QtBindings.NONE
)


def _set_qt_binding(qt_binding: QtBindings | str) -> str:
    """
    Check if qtpy can import `qt_binding` and return the qt_binding that will be used.
    Checks os.environ["QT_API"] first.
    If os.environ["QT_API"] is set it will take precedence over `qt_bindings`.
    Sets ``HAS_DEPENDENCY["qt"]`` to whether a binding could be loaded.

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

    HAS_DEPENDENCY["qt"] = bool(qt_api)
    return qt_api


def needs_package_version(
    package: str,
    major: str,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Function that returns a decorator to check for package.version starting with major.

    Parameters
    ----------
    package
        Package or dependency name to be checked.
    major
        Major version number that is required.

    Returns
    -------
    callable
        A decorator that raises RuntimeError if package is not available.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            version = importlib.metadata.version(package)
            if not version.startswith(major):  # type: ignore
                raise RuntimeError(
                    f"Function {func} needs {package}~{major}. "
                    f"The installed version is {package}={version}."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
