import pytest
from locan.dependencies import *


def test_IMPORT_NAMES():
    package = "numpy"
    assert IMPORT_NAMES.get(package, package) == "numpy"
    package = "scikit-learn"
    assert IMPORT_NAMES.get(package, package) == "sklearn"
    print(IMPORT_NAMES)


def test_dependency_lists():
    assert len(INSTALL_REQUIRES) > 0
    assert len(EXTRAS_REQUIRE) >= 0
    # print("install_requires:", INSTALL_REQUIRES)
    # print("extras_require:", EXTRAS_REQUIRE)


def test_has_dependency():
    with pytest.raises(KeyError):
        assert HAS_DEPENDENCY["numpy"]
    assert all(HAS_DEPENDENCY)


def test_needs_package():
    assert callable(needs_package("numpy"))


@pytest.mark.skipif(not HAS_DEPENDENCY["pytest"])
@needs_package("pytest")
def function_to_be_decorated():
    """This is documentation for function_to_be_decorated."""
    return True


def test_function_to_be_decorated():
    assert function_to_be_decorated.__doc__ == "This is documentation for function_to_be_decorated."
    assert function_to_be_decorated()
