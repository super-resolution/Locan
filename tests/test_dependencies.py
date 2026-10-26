import importlib.metadata
import logging

import pytest

from locan import (
    EXTRAS_REQUIRE,
    HAS_DEPENDENCY,
    IMPORT_NAMES,
    INSTALL_REQUIRES,
    QtBindings,
    needs_package,
    needs_package_version,
)
from locan.dependencies import (
    _get_dependencies,
    _has_dependency_factory,
    _import_name,
    _normalize_name,
)


def test_IMPORT_NAMES():
    package = "numpy"
    assert IMPORT_NAMES.get(package, package) == "numpy"
    package = "scikit-learn"
    assert IMPORT_NAMES.get(package, package) == "sklearn"


def test__get_dependencies():
    required_dependencies, extra_dependencies = _get_dependencies(package="locan")
    assert required_dependencies
    assert extra_dependencies
    # print(required_dependencies, extra_dependencies)
    required_dependencies, extra_dependencies = _get_dependencies(package="numpy")
    assert len(required_dependencies) == 0
    assert len(extra_dependencies) == 0


def test_dependency_lists():
    assert len(INSTALL_REQUIRES) > 0
    assert len(INSTALL_REQUIRES) == len(set(INSTALL_REQUIRES))
    assert len(EXTRAS_REQUIRE) >= 0
    # print("install_requires:", INSTALL_REQUIRES)
    # print("extras_require:", EXTRAS_REQUIRE)


def test_has_dependency():
    assert "numpy" in HAS_DEPENDENCY
    assert all(HAS_DEPENDENCY)


def test_needs_package():
    assert callable(needs_package("numpy"))


@needs_package("pytest")
def function_to_be_decorated():
    """This is documentation for function_to_be_decorated."""
    return True


def test_function_to_be_decorated():
    assert (
        function_to_be_decorated.__doc__
        == "This is documentation for function_to_be_decorated."
    )
    assert function_to_be_decorated()


def test_QtBindings():
    assert QtBindings.NONE == ""
    for item in QtBindings:
        if item == QtBindings.NONE:
            assert not QtBindings.NONE.value
        else:
            assert item.value


def test_needs_package_version():
    @needs_package_version(package="numpy", major="0")
    def my_func():
        return

    with pytest.raises(RuntimeError):
        my_func()


@pytest.mark.parametrize(
    "name, expected",
    [
        ("numpy", "numpy"),
        ("Mpl_Scatter.Density", "mpl-scatter-density"),
        ("PySide6", "pyside6"),
        ("open3d-cpu", "open3d-cpu"),
        ("tomli--w", "tomli-w"),
    ],
)
def test__normalize_name(name, expected):
    assert _normalize_name(name) == expected


@pytest.mark.parametrize(
    "package, expected",
    [
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("mpl-scatter-density", "mpl_scatter_density"),
        ("mpl_scatter_density", "mpl_scatter_density"),
        ("pyside6", "PySide6"),
        ("PySide6", "PySide6"),
        ("PyQt5", "PyQt5"),
        ("open3d-cpu", "open3d"),
        ("tomli-w", "tomli_w"),
        ("pydantic-settings", "pydantic_settings"),
    ],
)
def test__import_name(package, expected):
    assert _import_name(package) == expected


def test__import_name_custom_mapping():
    assert _import_name("my-package", import_names={"my-package": "mypkg"}) == "mypkg"
    # non-normalized keys of a user-supplied mapping are honoured as well
    assert _import_name("My_Package", import_names={"My_Package": "mypkg"}) == "mypkg"


def test__has_dependency_factory_keys():
    has_dependency = _has_dependency_factory(
        ["mpl_scatter_density", "open3d-cpu", "pyside6", "tomli-w", "numpy"]
    )
    assert set(has_dependency) == {
        "mpl_scatter_density",
        "open3d",
        "PySide6",
        "tomli_w",
        "numpy",
    }
    assert all(isinstance(value, bool) for value in has_dependency.values())
    assert has_dependency["numpy"] is True


def test_has_dependency_no_hyphen_keys():
    assert not any("-" in key for key in HAS_DEPENDENCY)
    assert "pyside6" not in HAS_DEPENDENCY
    assert "open3d-cpu" not in HAS_DEPENDENCY


def test_has_dependency_unknown_key(caplog):
    has_dependency = _has_dependency_factory([])
    with caplog.at_level(logging.WARNING, logger="locan.dependencies"):
        assert has_dependency["not_a_dependency_xyz"] is False
        assert has_dependency["not_a_dependency_xyz"] is False
    assert "not_a_dependency_xyz" not in has_dependency
    assert has_dependency.get("not_a_dependency_xyz", None) is None
    warnings = [
        record for record in caplog.records if "not_a_dependency_xyz" in record.message
    ]
    assert len(warnings) == 1


def test_has_dependency_not_installed(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "requires", lambda package: None)
    required, extras = _get_dependencies(package="locan")
    assert required == set()
    assert extras == set()
    has_dependency = _has_dependency_factory(required.union(extras))
    assert has_dependency["open3d"] is False
    assert has_dependency["napari"] is False


def test_needs_package_unknown_dependency():
    @needs_package("not_a_dependency_xyz")
    def my_func():
        return True

    with pytest.raises(ImportError, match="not_a_dependency_xyz"):
        my_func()


def test_has_dependency_qt():
    from locan.configuration import QT_BINDING

    assert "qt" in HAS_DEPENDENCY
    assert isinstance(HAS_DEPENDENCY["qt"], bool)
    assert HAS_DEPENDENCY["qt"] == bool(QT_BINDING)
