#!/usr/bin/env python

"""
Run project test suite with pytest.


To run the script::

    test

Try for instance::

    test

See Also
--------
surepy.tests.test
"""
import argparse

from surepy.tests import test as sc_test


def _add_arguments(parser):
    return


def main(args=None):

    parser = argparse.ArgumentParser(description='Run test suite.')
    _add_arguments(parser)
    returned_args = parser.parse_args(args)
    sc_test()


if __name__ == '__main__':
    main()
