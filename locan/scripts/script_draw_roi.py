#!/usr/bin/env python

"""
Define regions of interest with matplotlib and save as roi files.

With this script you can choose a file name, open the localization file and
draw a rectangular region of interest.
Within the matplotlib image draw a rectange, type '+' to add the roi to the
list, then type 'q' to quit.
The roi is then saved as _roi.yaml file.

To run the script::

    locan draw_roi_mpl -f <file> -t <file type> -i <roi file indicator> -r <region type>

Try for instance::

    locan draw_roi_mpl -f "locan/tests/test_data/five_blobs.txt" -t 1 -i "_roi" -r "ellipse"

See Also
--------
locan.rois.select_by_drawing_mpl

"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Literal

import locan.locan_io.locdata.io_locdata as io
from locan.constants import FileType
from locan.data import metadata_pb2
from locan.gui.io import file_dialog
from locan.visualize.render_mpl.render2d import select_by_drawing_mpl


def sc_draw_roi_mpl(
    file_path: str | os.PathLike[Any] | None = None,
    file_type: int | str | FileType | metadata_pb2.Metadata = 1,
    roi_file_indicator: str = "_roi",
    region_type: Literal["rectangle", "ellipse", "polygon"] = "rectangle",
) -> None:
    """
    Define regions of interest by drawing a boundary.

    Parameters
    ----------
    file_path
        File path to localization data.
    file_type
        Indicator for the file type.
        Integer or string should be according to locan.constants.FileType.
    roi_file_indicator
        Indicator to add to the localization file name and use as roi file
        name (with further extension .yaml).
    region_type
        rectangle, ellipse, or polygon specifying the selection widget to use.
    """
    # choose file interactively
    if file_path is None:
        file_path = Path(file_dialog(message="choose file", filter="*.txt; *.csv")[0])
    else:
        file_path = Path(file_path)

    print(file_path)

    # load data
    dat = io.load_locdata(path=file_path, file_type=file_type)

    # set roi
    rois = select_by_drawing_mpl(
        locdata=dat, bin_size=50, rescale="equal", region_type=region_type
    )
    print(rois)

    # save roi
    for i, roi in enumerate(rois):
        roi_file = file_path.stem + roi_file_indicator + f"_{i}.yaml"
        roi_path = file_path.with_name(roi_file)
        roi.to_yaml(path=roi_path)


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
        default=1,
        help="Integer or string indicating the file type.",
    )
    parser.add_argument(
        "-i",
        "--indicator",
        dest="roi_file_indicator",
        type=str,
        default="_roi",
        help="Indicator to add to the localization file name and use as roi file name "
        "(with further extension .yaml).",
    )
    parser.add_argument(
        "-r",
        "--region",
        dest="region",
        type=str,
        default="rectangle",
        help="String indicating the region type.",
    )


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Set roi by drawing a boundary.")
    _add_arguments(parser)
    returned_args = parser.parse_args(args)

    sc_draw_roi_mpl(
        returned_args.file,
        returned_args.type,
        returned_args.roi_file_indicator,
        returned_args.region_type,
    )


if __name__ == "__main__":
    main()
