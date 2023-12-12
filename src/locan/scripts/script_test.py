#!/usr/bin/env python

"""
Run project test suite with pytest.


To run the script::

    test

Try for instance::

    test

See Also
--------
locan.tests.test
"""
from __future__ import annotations

import argparse

from locan.tests import test as sc_test


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "pytest_args",
        nargs="?",
        type=str,
        default=None,
        help="string with pytest options.",
    )


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run test suite.")
    _add_arguments(parser)
    returned_args = parser.parse_args(args)
    sc_test(args=returned_args.pytest_args)


if __name__ == "__main__":
    main()
