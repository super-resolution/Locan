"""
Utility script to print locan.__all__ elements.

These items need to be explicitly re-exported in locan.__init__
to satisfy mypy.
"""
from importlib import import_module


def main() -> None:
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
    for submodule in submodules:
        module = import_module(name=f".{submodule}", package="locan")
        lines = ",\n".join([f"{item} as {item}" for item in sorted(module.__all__)])
        text = (
            f"from {module.__name__} import (  # type:ignore[attr-defined]\n"
            + lines
            + "\n)"
        )
        print(text)


if __name__ == "__main__":
    main()
