"""
Utility script to print locan.__all__ elements.

These items need to be explicitly re-exported in locan.__init__
to satisfy mypy.
"""
from importlib import import_module

submodules: list[str] = [
    "analysis",
    "configuration",
    "constants",
    "data",
    "datasets",
    "dependencies",
    "gui",
    "locan_io",
    "simulation",
    "tests",
    "utils",
    "visualize",
]


def make_init(submodule: str, package: str) -> str:
    """
    Replace
    'from .submodule import *'
    with
    'from .submodule import (<explicit list of functions and classes>)'
    according to provided __all__.

    The output will be copied into submodule.__init__.py
    or submodule.__init__.pyi.

    Parameters
    ----------
    submodule
        the module from which to import
    package
        the anchor point from which to resolve relative submodule specifications.

    Returns
    -------
    str
    """
    module = import_module(name=f".{submodule}", package=package)
    lines = ",\n".join([f"{item} as {item}" for item in sorted(module.__all__)])
    relative_submodule = ".".join(module.__name__.split("locan."))
    text = f"from {relative_submodule} import (\n" + lines + "\n)" + "\n"
    return text


def make_explicit_import_command(submodules: list[str], package: str) -> str:
    """
    Print explicitly re-exported imports from submodules
    'from .submodule import (<explicit list of functions and classes>)'
    according to provided __all__.

    The output will be copied into submodule.__init__.py
    or submodule.__init__.pyi.

    Parameters
    ----------
    submodules
        the module from which to import
    package
        the anchor point from which to resolve relative submodule specifications.

    Returns
    -------
    str
    """
    text = ""
    for submodule in submodules:
        module = import_module(name=f".{submodule}", package=package)
        lines = ",\n".join([f"{item} as {item}" for item in sorted(module.__all__)])
        relative_submodule = ".".join(module.__name__.split("locan."))
        text += f"from {relative_submodule} import (\n" + lines + "\n)" + "\n"
    return text


def make_explicit_all(submodules: list[str], package: str) -> str:
    """
    Print explicitly __all__
    according to provided submodules.__all__.

    The output will be copied into submodule.__init__.py
    or submodule.__init__.pyi.

    Parameters
    ----------
    submodules
        the module from which to import
    package
        the anchor point from which to resolve relative submodule specifications.

    Returns
    -------
    str
    """
    all_ = []
    for submodule in submodules:
        module = import_module(name=f".{submodule}", package="locan")
        all_.extend(module.__all__)
    text = "__all__ = [\n'" + "',\n'".join(sorted(all_)) + "',\n]" + "\n"
    return text


if __name__ == "__main__":
    package = "locan"
    # print(submodules)

    print(make_init(submodule="data.cluster", package=package))

    # print(make_explicit_all(submodules=submodules, package=package))
    # print(make_explicit_import_command(submodules=submodules, package=package))
