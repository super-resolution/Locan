"""
command-line interface
"""
from __future__ import annotations

import argparse
import sys

from locan import __version__
from locan.scripts.script_check import _add_arguments as _add_arguments_check
from locan.scripts.script_draw_roi import _add_arguments as _add_arguments_draw_roi
from locan.scripts.script_napari import _add_arguments as _add_arguments_napari
from locan.scripts.script_rois import _add_arguments as _add_arguments_rois
from locan.scripts.script_show_versions import (
    _add_arguments as _add_arguments_show_versions,
)
from locan.scripts.script_test import _add_arguments as _add_arguments_test
from locan.utils.system_information import show_versions


def main(args: list[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Entry point for locan.")

    parser.add_argument(
        "--version",
        action="version",
        version=f"locan version {__version__}",
        help="show version and exit.",
    )

    parser.add_argument(
        "--info", action="store_true", help="show system information and exit."
    )

    subparsers = parser.add_subparsers(dest="command")

    # parser for the command draw_roi_mpl
    parser_draw_roi = subparsers.add_parser(
        name="draw_roi_mpl", description="Set roi by drawing a boundary in mpl."
    )
    _add_arguments_draw_roi(parser_draw_roi)

    # parser for the command rois
    parser_rois = subparsers.add_parser(
        name="rois", description="Define rois by adding shapes in napari."
    )
    _add_arguments_rois(parser_rois)

    # parser for the command check
    parser_check = subparsers.add_parser(
        name="check", description="Show localizations in original recording."
    )
    _add_arguments_check(parser_check)

    # parser for the command napari
    parser_napari = subparsers.add_parser(
        name="napari", description="Render localization data in napari."
    )
    _add_arguments_napari(parser_napari)

    # parser for the command show_versions
    parser_show_versions = subparsers.add_parser(
        name="show_versions",
        description="Show system information and dependency versions.",
    )
    _add_arguments_show_versions(parser_show_versions)

    # parser for the command test
    parser_test = subparsers.add_parser(name="test", description="Run test suite.")
    _add_arguments_test(parser_test)

    # Parse
    returned_args = parser.parse_args(args)

    if returned_args.info:
        show_versions()

    elif returned_args.command:
        if returned_args.command == "draw_roi_mpl":
            from .scripts.script_draw_roi import sc_draw_roi_mpl

            sc_draw_roi_mpl(
                returned_args.file,
                returned_args.type,
                returned_args.roi_file_indicator,
                returned_args.region_type,
            )

        elif returned_args.command == "rois":
            from .scripts.script_rois import sc_draw_roi_napari

            sc_draw_roi_napari(
                file_path=returned_args.file,
                file_type=returned_args.type,
                roi_file_indicator=returned_args.roi_file_indicator,
                bin_size=returned_args.bin_size,
                rescale=returned_args.rescale,
            )

        elif returned_args.command == "check":
            from .scripts.script_check import sc_check

            sc_check(
                pixel_size=returned_args.pixel_size,
                file_images=returned_args.file_images,
                file_locdata=returned_args.file_locdata,
                file_type=returned_args.file_type,
                transpose=True,
                kwargs_image={},
                kwargs_points={},
            )

        elif returned_args.command == "napari":
            from .scripts.script_napari import sc_napari

            sc_napari(
                file_path=returned_args.file,
                file_type=returned_args.type,
                bin_size=returned_args.bin_size,
                rescale=returned_args.rescale,
            )

        elif returned_args.command == "show_versions":
            from .scripts.script_show_versions import sc_show_versions  # type: ignore[attr-defined]  # noqa

            sc_show_versions(
                verbose=returned_args.verbose,
                extra_dependencies=returned_args.extra,
                other_dependencies=returned_args.other,
            )

        elif returned_args.command == "test":
            from .scripts.script_test import sc_test  # type: ignore[attr-defined]

            sc_test(args=returned_args.pytest_args)

    else:
        print(
            "This is the command line entry point for locan. Get more information with 'locan -h'."
        )

    return None


if __name__ == "__main__":
    sys.exit(main())  # type: ignore[func-returns-value]
