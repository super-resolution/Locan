#!/usr/bin/env python

"""
Define regions of interest with napari and save as roi files.

With this script you can choose a file name, open the localization file in napari.
Draw regions of interest as additional shapes in napari.
Upon closing napari each shape is taken as single roi and saved as _roi.yaml file.

To run the script::

    rois -f <file> -t <file type> -i <roi file indicator>

Try for instance::

    rois -f "surepy/tests/test_data/five_blobs.txt" -t 1 -i "_roi"

See Also
--------
surepy.data.rois.select_by_drawing_napari
"""
import argparse
from pathlib import Path

from surepy.constants import _has_napari
if _has_napari: import napari

from surepy.constants import FileType
from surepy.gui.io import file_dialog
import surepy.io.io_locdata as io
from surepy.render.render2d import select_by_drawing_napari


def sc_draw_roi_napari(file_path=None, file_type=FileType.CUSTOM, roi_file_indicator='_roi', bin_size=50):
    """
    Define regions of interest by drawing a boundary.

    Parameters
    ----------
    file_path : string or os.PathLike
        File path to localization data.
    file_type : int, str, surepy.constants.FileType, metadata_pb2
        Indicator for the file type.
        Integer or string should be according to surepy.constants.FileType.
    roi_file_indicator : str
        Indicator to add to the localization file name and use as roi file name (with further extension .yaml).
    """
    with napari.gui_qt():
        # choose file interactively
        if file_path is None:
            file_path = Path(file_dialog(message='choose file', filter='*.txt; *.csv')[0])
        else:
            file_path = Path(file_path)

        print(file_path)

        # load data
        dat = io.load_locdata(path=file_path, file_type=file_type)

        # set roi
        rois = select_by_drawing_napari(locdata=dat, bin_size=bin_size, rescale='equal')
        print(rois)

        # save roi
        for i, roi in enumerate(rois):
            roi_file = file_path.stem + roi_file_indicator + f'_{i}.yaml'
            roi_path = file_path.with_name(roi_file)
            roi.to_yaml(path=roi_path)


def _add_arguments(parser):
    parser.add_argument('-f', '--file', dest='file', type=str, default=None,
                        help='File path to localization data.')
    parser.add_argument('-t', '--type', dest='type', type=int, default=2,
                        help='Integer or string indicating the file type.')
    parser.add_argument('-i', '--indicator', dest='roi_file_indicator', type=str, default='_roi',
                        help='Indicator to add to the localization file name and use as roi file name '
                             '(with further extension .yaml).')
    parser.add_argument('--bin_size', dest='bin_size', type=float, default=50,
                        help='Keyword passed to render function.')


def main(args=None):

    parser = argparse.ArgumentParser(description='Set roi by drawing a boundary.')
    _add_arguments(parser)
    returned_args = parser.parse_args(args)

    sc_draw_roi_napari(returned_args.file, returned_args.type, returned_args.roi_file_indicator,
                       bin_size=returned_args.bin_size)


if __name__ == '__main__':
    main()
