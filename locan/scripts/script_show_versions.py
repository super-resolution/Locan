#!/usr/bin/env python

"""
Show system information and dependency versions.


To run the script::

    show_versions -v -e -o <module name> [<module name>...]

Try for instance::

    show_versions

See Also
--------
locan.utils.system_information.show_versions
"""
import argparse

from locan.utils.system_information import show_versions as sc_show_versions


def _add_arguments(parser):
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Include information on node and executable path.')
    parser.add_argument('-e', '--extra', dest='extra', action='store_true',
                        help='Include extra dependencies as specified in setup.py.')
    parser.add_argument('-o', '--other', dest='other', type=str, nargs='*',
                        help='Include other module names.')


def main(args=None):

    parser = argparse.ArgumentParser(description='Show system information and dependency versions.')
    _add_arguments(parser)
    returned_args = parser.parse_args(args)
    print(returned_args)
    sc_show_versions(verbose=returned_args.verbose, extra_dependencies=returned_args.extra,
                  other_dependencies=returned_args.other)


if __name__ == '__main__':
    main()
