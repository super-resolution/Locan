"""
surepy command line interface
"""
import sys
import argparse


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Entry point for surepy.')

    subparsers = parser.add_subparsers(dest='command')

    # parser for the command_1
    parser_1 = subparsers.add_parser('command_1')
    parser_1.add_argument('-x', type=int, default=1, help='something')
    parser_1.add_argument('y', type=float)
    # parser_1.set_defaults(func=command_1)

    # Parse
    returned_args = parser.parse_args(args)

    if returned_args.command:
        if returned_args.command == "command_1":
            from .scripts.command_1 import command_1
            command_1()

    else:
        print("This is the command line entry point for surepy.")


if __name__ == "__main__":
    main()
