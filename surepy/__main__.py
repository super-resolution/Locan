"""
command-line interface
"""
import sys
import argparse

from surepy.scripts.draw_roi import _add_arguments as _add_arguments_draw_roi


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Entry point for surepy.')

    subparsers = parser.add_subparsers(dest='command')

    # parser for the command_1
    parser_command_1 = subparsers.add_parser('command_1')
    parser_command_1.add_argument('-x', type=int, default=1, help='something')
    parser_command_1.add_argument('y', type=float)
    # parser_command_1.set_defaults(func=command_1)

    # parser for the command roi
    parser_roi = subparsers.add_parser(name='roi',
                                       description='Set roi by drawing a boundary.')
    _add_arguments_draw_roi(parser_roi)

    # Parse
    returned_args = parser.parse_args(args)

    if returned_args.command:
        if returned_args.command == "command_1":
            from .scripts.command_1 import command_1
            command_1()

        elif returned_args.command == "roi":
            from .scripts.draw_roi import draw_roi
            draw_roi(returned_args.directory, returned_args.type, returned_args.roi_file_indicator,
                     returned_args.region_type)

    else:
        print("This is the command line entry point for surepy.")


if __name__ == "__main__":
    sys.exit(main())
