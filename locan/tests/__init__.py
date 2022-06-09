"""

Test the locan package.

This module includes unit tests for all modules within the locan package.
The tests are organized following the subpackage structure of locan.

"""
from locan import ROOT_DIR

__all__ = ["test"]


def test(args=None):
    """
    Running tests with pytest.

    Parameters
    ----------
    args : str, list of str
        Parameters passed to :func:`pytest.main`
    """
    try:
        import pytest
    except ImportError:
        raise ImportError("Need pytest to run tests.")

    extra_args = ["-m not gui and not visual and not requires_datasets"]

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
