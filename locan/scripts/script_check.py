#!/usr/bin/env python

"""
Show original SMLM images overlaid with localization data.
Data is rendered in napari.

To run the script::

    locan check <pixel size> -f <images file> -l <localization file> -t <file type>

Try for instance::

    locan check 133 -f "locan/tests/test_data/images.tif" -l "locan/tests/test_data/rapidStorm_from_images.txt" -t 2
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import tifffile as tif

from locan import locdata_id
from locan.constants import FileType
from locan.data import metadata_pb2
from locan.data.locdata import LocData
from locan.dependencies import HAS_DEPENDENCY
from locan.gui import file_dialog
from locan.locan_io.locdata.io_locdata import load_locdata

if HAS_DEPENDENCY["napari"]:
    import napari


def render_locs_per_frame_napari(
    images: npt.ArrayLike,
    pixel_size: float | tuple[float],
    locdata: LocData,
    viewer: napari.Viewer = None,
    transpose: bool = True,
    kwargs_image: dict[str, Any] | None = None,
    kwargs_points: dict[str, Any] | None = None,
) -> napari.Viewer:
    """
    Display original recording and overlay localization spots in napari.

    Parameters
    ---------
    images
        Stack of raw data as recorded by camera.
    pixel_size
        Pixel size for images (in locdata units) with shape (2,).
    transpose
        If True transpose x and y axis of `images`.
    locdata
        Localization data that corresponds to `images` raw data.
    viewer
        The viewer object on which to add the image
    kwargs_image
        Other parameters passed to napari.Viewer().add_image().
    kwargs_points : dict
        Other parameters passed to napari.Viewer().add_points().

    Returns
    -------
    napari.Viewer
        Viewer with the image.
    """
    if kwargs_image is None:
        kwargs_image = {}
    if kwargs_points is None:
        kwargs_points = {}

    if np.ndim(pixel_size) == 0:
        pixel_size_ = (pixel_size, pixel_size)
    elif np.ndim(pixel_size) == 1 and len(pixel_size) == 2:  # type:  ignore[arg-type]
        pixel_size_ = pixel_size  # type: ignore[assignment]
    else:
        raise TypeError("Dimension of `pixel_size` is incompatible with 2d image.")

    # transpose x and y axis
    if transpose:
        images_ = np.transpose(images, (0, 2, 1))
    else:
        images_ = images  # type: ignore[assignment]

    points = locdata.data[
        locdata.data["frame"] < len(images)  # type:  ignore[arg-type]
    ][["frame", "position_x", "position_y"]].values

    # Provide napari viewer if not provided
    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(images_, name="Raw data", **kwargs_image, scale=pixel_size_)
    viewer.add_points(
        data=points,
        name=f"LocData {locdata_id}",
        symbol="disc",
        size=500,
        face_color="r",
        edge_color="r",
        opacity=0.3,
        **kwargs_points,
    )

    return viewer


def sc_check(
    pixel_size: float | tuple[float],
    file_images: str | os.PathLike[Any] | None = None,
    file_locdata: str | os.PathLike[Any] | None = None,
    file_type: int | str | FileType | metadata_pb2.Metadata = FileType.RAPIDSTORM,
    viewer: napari.Viewer = None,
    transpose: bool = True,
    kwargs_image: dict[str, Any] | None = None,
    kwargs_points: dict[str, Any] | None = None,
) -> None:
    """
    Load and display original recording and load and overlay localization spots in napari.

    Parameters
    ---------
    pixel_size
        Pixel size for images (in locdata units) with shape (2,).
    file_images
        File path for stack of raw data as recorded by camera.
    file_locdata
        File path for localization data that corresponds to `images` raw data.
    file_type
        Indicator for the file type.
        Integer or string should be according to locan.constants.FileType.
    transpose
        If True transpose x and y axis of `images`.
    viewer : napari.Viewer
        The viewer object on which to add the image
    kwargs_image : dict
        Other parameters passed to :meth:`napari.Viewer.add_image`.
    kwargs_points : dict
        Other parameters passed to :meth:`napari.Viewer.add_points`.
    """
    if kwargs_image is None:
        kwargs_image = {}
    if kwargs_points is None:
        kwargs_points = {}

    with napari.gui_qt():
        # load images
        if file_images is None:
            file_images = file_dialog(
                directory=None,
                message="Select images file...",
                filter="Tif files (*.tif)",
            )[0]
        # load images from tiff file
        image_stack = tif.imread(
            str(file_images)
        )  # , key=range(0, 10, 1))  - maybe add key parameter

        # load locdata
        if file_locdata is None:
            file_locdata = file_dialog(
                directory=None,
                message="Select a localization file...",
                filter="Text files (*.txt);; CSV files (*.csv)",
            )[0]
        locdata = load_locdata(file_locdata, file_type=file_type)

        # due to changed napari behavior from v3.0 on the context manager is moved up.
        # with napari.gui_qt():
        render_locs_per_frame_napari(
            image_stack,  # type:  ignore[arg-type]
            pixel_size,
            locdata,
            viewer,
            transpose,
            kwargs_image,
            kwargs_points,
        )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        dest="pixel_size", type=float, help="Pixel size for images (in locdata units)."
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="file_images",
        type=str,
        default=None,
        help="File with images of original recording.",
    )
    parser.add_argument(
        "-l",
        "--localizations",
        dest="file_locdata",
        type=str,
        default=None,
        help="File with localization data.",
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="file_type",
        type=int,
        default=2,
        help="Integer or string indicating the file type.",
    )


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Show localizations in original recording."
    )
    _add_arguments(parser)
    returned_args = parser.parse_args(args)

    sc_check(
        pixel_size=returned_args.pixel_size,
        file_images=returned_args.file_images,
        file_locdata=returned_args.file_locdata,
        file_type=returned_args.file_type,
        transpose=True,
        kwargs_image={},
        kwargs_points={},
    )


if __name__ == "__main__":
    main()
