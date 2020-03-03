"""
command-line interface
"""
import sys
import argparse

from surepy.scripts.script_draw_roi import _add_arguments as _add_arguments_draw_roi
from surepy.scripts.script_check import _add_arguments as _add_arguments_check
from surepy.scripts.script_rois import _add_arguments as _add_arguments_rois
from surepy.scripts.script_napari import _add_arguments as _add_arguments_napari
from surepy.scripts.script_show_versions import _add_arguments as _add_arguments_show_versions


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Entry point for surepy.')

    subparsers = parser.add_subparsers(dest='command')

    # parser for the command sc_draw_roi_mpl
    parser_draw_roi = subparsers.add_parser(name='sc_draw_roi_mpl',
                                       description='Set roi by drawing a boundary in mpl.')
    _add_arguments_draw_roi(parser_draw_roi)

    # parser for the command rois
    parser_rois = subparsers.add_parser(name='rois',
                                       description='Define rois by adding shapes in napari.')
    _add_arguments_rois(parser_rois)

    # parser for the command sc_check
    parser_check = subparsers.add_parser(name='sc_check',
                                       description='Show localizations in original recording.')
    _add_arguments_check(parser_check)

    # parser for the command napari
    parser_napari = subparsers.add_parser(name='napari',
                                       description='Render localization data in napari.')
    _add_arguments_napari(parser_napari)

    # parser for the command show_versions
    parser_show_versions = subparsers.add_parser(name='show_versions',
                                       description='Show system information and dependency versions.')
    _add_arguments_show_versions(parser_show_versions)

    # Parse
    returned_args = parser.parse_args(args)

    if returned_args.command:

        if returned_args.command == "sc_draw_roi_mpl":
            from .scripts.script_draw_roi import sc_draw_roi_mpl
            sc_draw_roi_mpl(returned_args.file, returned_args.type, returned_args.roi_file_indicator,
                            returned_args.region_type)

        elif returned_args.command == "rois":
            from .scripts.script_rois import sc_draw_roi_napari
            sc_draw_roi_napari(returned_args.file, returned_args.type, returned_args.roi_file_indicator)

        elif returned_args.command == "sc_check":
            from .scripts.script_check import sc_check
            sc_check(pixel_size=returned_args.pixel_size, file_images=returned_args.file_images,
                     file_locdata=returned_args.file_locdata, file_type=returned_args.file_type,
                     transpose=True, kwargs_image={}, kwargs_points={})

        elif returned_args.command == "napari":
            from .scripts.script_napari import sc_napari
            sc_napari(returned_args.file, returned_args.type, bin_size=returned_args.bin_size)

        elif returned_args.command == "show_versions":
            from .scripts.script_show_versions import show_versions
            show_versions(verbose=returned_args.verbose, extra_dependencies=returned_args.extra,
                          other_dependencies=returned_args.other)

    else:
        print("This is the command line entry point for surepy.")


if __name__ == "__main__":
    sys.exit(main())
