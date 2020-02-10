"""

Test the surepy package.

This module includes unit tests for all modules within the surepy package.
The tests are organized following the subpackage structure of surepy.

"""
from surepy.constants import ROOT_DIR

__all__ = ['test']


def test(args=None):
    """
    Running tests with pytest.

    Parameters
    ----------
    args : string or list of string
        Parameters passed to pytest.main()
    """
    try:
        import pytest
    except ImportError:
        raise ImportError("Need pytest to run tests.")

    if args is None:
        extra_args = []
    elif not isinstance(args, list):
        extra_args = [args]
    else:
        extra_args = args

    test_directory = ROOT_DIR.joinpath('tests').as_posix()
    extra_args.append(test_directory)
    print(f'running: pytest {" ".join(extra_args)}')
    return pytest.main(extra_args)
