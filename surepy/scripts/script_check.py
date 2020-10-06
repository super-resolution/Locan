#!/usr/bin/env python

"""
Show original SMLM images overlaid with localization data.
Data is rendered in napari.

To run the script::

    check <pixel size> -f <images file> -l <localization file> -t <file type>

Try for instance::

    check 130 -f "surepy/tests/test_data/images.tif" -l "surepy/tests/test_data/rapidStorm.txt" -t 2
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile as tif

from surepy.constants import _has_napari
if _has_napari: import napari

import surepy as sp


def render_locs_per_frame_napari(images, pixel_size, locdata, viewer=None, transpose=True,
                                 kwargs_image={}, kwargs_points={}):
    """
    Display original recording and overlay localization spots in napari.

    Parameters
    ---------
    images : np.array
        Stack of raw data as recorded by camera.
    pixel_size : float or tuple of float with shape (2,)
        Pixel size for images (in locdata units).
    transpose : bool
        If True transpose x and y axis of `images`.
    locdata : LocData object
        Localization data that corresponds to `images` raw data.
    viewer : napari viewer
        The viewer object on which to add the image

    Other Parameters
    ----------------
    kwargs_image : dict
        Other parameters passed to napari.Viewer().add_image().
    kwargs_points : dict
        Other parameters passed to napari.Viewer().add_points().

    Returns
    -------
    napari Viewer object
        Viewer with the image.
    """
    if np.ndim(pixel_size) == 0:
        pixel_size_ = (pixel_size, pixel_size)
    elif np.ndim(pixel_size) == 1 and len(pixel_size) == 2:
        pixel_size_ = pixel_size
    else:
        raise TypeError('Dimension of `pixel_size` is incompatible with 2d image.')

    # transpose x and y axis
    if transpose:
        images_ = np.transpose(images, (0, 2, 1))
    else:
        images_ = images

    points = locdata.data[locdata.data['frame']<len(images)][['frame', 'position_x', 'position_y']].values

    # Provide napari viewer if not provided
    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(images_, name=f'Raw data', **kwargs_image, scale=pixel_size_)
    viewer.add_points(data=points, name=f'LocData {sp.LOCDATA_ID}',
                      symbol='disc', size=500, face_color='r', edge_color='r', opacity=0.3,
                      **kwargs_points)

    return viewer


def sc_check(pixel_size, file_images=None, file_locdata=None, file_type=sp.FileType.RAPIDSTORM,
             viewer=None, transpose=True, kwargs_image={}, kwargs_points={}):
    """
    Load and display original recording and load and overlay localization spots in napari.

    Parameters
    ---------
    pixel_size : float or tuple of float with shape (2,)
        Pixel size for images (in locdata units).
    file_images : str or Path
        File path for stack of raw data as recorded by camera.
    file_locdata : str or Path
        File path for localization data that corresponds to `images` raw data.
    file_type : int, str, surepy.constants.FileType, metadata_pb2
        Indicator for the file type.
        Integer or string should be according to surepy.constants.FileType.
    transpose : bool
        If True transpose x and y axis of `images`.
    locdata : LocData object
        Localization data that corresponds to `images` raw data.
    viewer : napari viewer
        The viewer object on which to add the image

    Other Parameters
    ----------------
    kwargs_image : dict
        Other parameters passed to napari.Viewer().add_image().
    kwargs_points : dict
        Other parameters passed to napari.Viewer().add_points().

    Returns
    -------
    napari Viewer object
        Viewer with the image.
    """
    # load images
    if file_images is None:
        file_images = sp.file_dialog(directory=None,
                                     message='Select images file...', filter='Tif files (*.tif)')[0]
    # load images from tiff file
    image_stack = tif.imread(str(file_images))  #, key=range(0, 10, 1))  - maybe add key parameter

    # load locdata
    if file_locdata is None:
        file_locdata = sp.file_dialog(directory=None,
                                      message='Select a localization file...',
                                      filter='Text files (*.txt);; CSV files (*.csv)')[0]
    locdata = sp.load_locdata(file_locdata, file_type=file_type)

    with napari.gui_qt():
        render_locs_per_frame_napari(image_stack, pixel_size, locdata, viewer, transpose,
                                     kwargs_image, kwargs_points)


def _add_arguments(parser):
    parser.add_argument(dest='pixel_size', type=float,
                        help='Pixel size for images (in locdata units).')
    parser.add_argument('-f', '--file', dest='file_images', type=str, default=None,
                        help='File with images of original recording.')
    parser.add_argument('-l', '--localizations', dest='file_locdata', type=str, default=None,
                        help='File with localization data.')
    parser.add_argument('-t', '--type', dest='file_type', type=int, default=2,
                        help='Integer or string indicating the file type.')


def main(args=None):

    parser = argparse.ArgumentParser(description='Show localizations in original recording.')
    _add_arguments(parser)
    returned_args = parser.parse_args(args)

    sc_check(pixel_size=returned_args.pixel_size, file_images=returned_args.file_images,
             file_locdata=returned_args.file_locdata, file_type=returned_args.file_type,
             transpose=True, kwargs_image={}, kwargs_points={})


if __name__ == '__main__':
    main()
