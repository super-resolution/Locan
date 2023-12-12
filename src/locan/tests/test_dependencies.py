from locan import (
    EXTRAS_REQUIRE,
    HAS_DEPENDENCY,
    IMPORT_NAMES,
    INSTALL_REQUIRES,
    QtBindings,
    needs_package,
)
from locan.dependencies import _get_dependencies


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
    assert QtBindings.NONE
    for item in QtBindings:
        if item == QtBindings.NONE:
            assert not QtBindings.NONE.value
        else:
            assert QtBindings.PYQT5.value
