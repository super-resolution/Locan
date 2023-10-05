"""

Test the locan package.

This module includes unit tests for all modules within the locan package.
The tests are organized following the subpackage structure of locan.

"""
from __future__ import annotations

from typing import TYPE_CHECKING

from locan import ROOT_DIR

if TYPE_CHECKING:
    from pytest import ExitCode

__all__: list[str] = ["test"]


def test(args: str | list[str] | None = None) -> int | ExitCode:
    """
    Running tests with pytest.

    Parameters
    ----------
    args
        Parameters passed to :func:`pytest.main`
    """
    try:
        import pytest
    except ImportError as exc:
        raise ImportError("Need pytest to run tests.") from exc

    extra_args = ["-m not gui and not visual"]

    if args is None:
        pass  # extra_args = []
    elif isinstance(args, list):
        extra_args.extend(args)  # extra_args = args
    else:
        extra_args.append(args)  # extra_args = [args]

    test_directory = ROOT_DIR.joinpath("tests").as_posix()
    extra_args.append(test_directory)
    print(f'running: pytest {" ".join(extra_args)}')
    return pytest.main(extra_args)
