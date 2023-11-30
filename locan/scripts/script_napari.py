#!/usr/bin/env python

"""
Render localization data in napari.

With this script you can choose a file name and render the localization file in napari.

To run the script::

    locan napari -f <file> -t <file type> -k <string with kwargs for render function> --bin_size <>
    --rescale <string with tuple or rescale>

Try for instance::

    locan napari -f "locan/tests/test_data/five_blobs.txt" -t 1 --bin_size 10 --rescale "0 1"

See Also
--------
locan.render.render2d.render_2d_napari
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import locan.locan_io.locdata.io_locdata as io
from locan.constants import FileType
from locan.data import metadata_pb2
from locan.dependencies import HAS_DEPENDENCY
from locan.gui.io import file_dialog
from locan.scripts.utilities import _type_converter_rescale
from locan.visualize.render_napari.render2d import render_2d_napari

if HAS_DEPENDENCY["napari"]:
    import napari


def sc_napari(
    file_path: str | os.PathLike[Any] | None = None,
    file_type: int | str | FileType | metadata_pb2.Metadata = FileType.CUSTOM,
    **kwargs: Any,
) -> None:
    """
    Render localization data in napari.

    Parameters
    ----------
    file_path
        File path to localization data.
    file_type
        Indicator for the file type.
        Integer or string should be according to locan.constants.FileType.
    kwargs
        Other parameters passed to :func:`render_2d_napari`.
    """
    # choose file interactively
    if file_path is None:
        file_path = Path(
            file_dialog(
                message="choose file",
                filter="Text files (*.txt);; CSV files (*.csv);; All files (*);; All files ()",
            )[0]
        )
    else:
        file_path = Path(file_path)

    print(file_path)

    # load data
    dat = io.load_locdata(path=file_path, file_type=file_type)

    # render
    render_2d_napari(locdata=dat, **kwargs)
    napari.run()


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        type=str,
        default=None,
        help="File path to localization data.",
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="type",
        type=int,
        default=2,
        help="Integer or string indicating the file type.",
    )
    parser.add_argument(
        "--bin_size",
        dest="bin_size",
        type=float,
        default=10,
        help="The size of bins in units of locdata coordinate units. "
        "Keyword passed to render_2d_napari function.",
    )
    parser.add_argument(
        "--rescale",
        dest="rescale",
        type=_type_converter_rescale,
        default="EQUALIZE",
        help="Rescale intensity values. Keyword passed to render_2d_napari function.",
    )


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render localization data in napari.")
    _add_arguments(parser)
    returned_args = parser.parse_args(args)

    sc_napari(
        returned_args.file,
        returned_args.type,
        bin_size=returned_args.bin_size,
        rescale=returned_args.rescale,
    )


if __name__ == "__main__":
    main()
