"""
command-line interface
"""
import sys
import argparse

from surepy.scripts.draw_roi import _add_arguments as _add_arguments_draw_roi
from surepy.scripts.check import _add_arguments as _add_arguments_check
from surepy.scripts.rois import _add_arguments as _add_arguments_rois


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

    # parser for the command draw_roi_mpl
    parser_draw_roi = subparsers.add_parser(name='draw_roi_mpl',
                                       description='Set roi by drawing a boundary in mpl.')
    _add_arguments_draw_roi(parser_draw_roi)

    # parser for the command rois
    parser_rois = subparsers.add_parser(name='rois',
                                       description='Define rois by adding shapes in napari.')
    _add_arguments_rois(parser_rois)

    # parser for the command check
    parser_check = subparsers.add_parser(name='check',
                                       description='Show localizations in original recording.')
    _add_arguments_check(parser_check)

    # Parse
    returned_args = parser.parse_args(args)

    if returned_args.command:
        if returned_args.command == "command_1":
            from .scripts.command_1 import command_1
            command_1()

        elif returned_args.command == "draw_roi_mpl":
            from .scripts.draw_roi import draw_roi_mpl
            draw_roi_mpl(returned_args.directory, returned_args.type, returned_args.roi_file_indicator,
                         returned_args.region_type)

        elif returned_args.command == "rois":
            from .scripts.rois import draw_roi_napari
            draw_roi_napari(returned_args.directory, returned_args.type, returned_args.roi_file_indicator)

        elif returned_args.command == "check":
            from .scripts.check import check_napari
            check_napari(pixel_size=returned_args.pixel_size, file_images=returned_args.file_images,
                         file_locdata=returned_args.file_locdata, file_type=returned_args.file_type,
                         transpose=True, kwargs_image={}, kwargs_points={})

    else:
        print("This is the command line entry point for surepy.")


if __name__ == "__main__":
    sys.exit(main())
