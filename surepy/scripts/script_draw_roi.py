#!/usr/bin/env python

"""
Define regions of interest with matplotlib and save as roi files.

With this script you can choose a file name, open the localization file and draw a rectangular region of interest.
Within the matplotlib image draw a rectange, type '+' to add the roi to the list, then type 'q' to quit.
The roi is then saved as _roi.yaml file.

To run the script::

    draw_roi_mpl -f <file> -t <file type> -i <roi file indicator> -r <region type>

Try for instance::

    draw_roi_mpl -f "surepy/tests/test_data/five_blobs.txt" -t 1 -i "_roi" -r "ellipse"

See Also
--------
surepy.data.rois.select_by_drawing_mpl

"""

import argparse
from pathlib import Path

from surepy.gui.io import file_dialog
import surepy.io.io_locdata as io
from surepy.render.render2d import select_by_drawing_mpl


def sc_draw_roi_mpl(file_path=None, file_type=1, roi_file_indicator='_roi', region_type='rectangle'):
    """
    Define regions of interest by drawing a boundary.

    Parameters
    ----------
    file_path : str, os.PathLike
        File path to localization data.
    file_type : int, str, surepy.constants.FileType, surepy.data.metadata_pb2.Metadata
        Indicator for the file type.
        Integer or string should be according to surepy.constants.FileType.
    roi_file_indicator : str
        Indicator to add to the localization file name and use as roi file name (with further extension .yaml).
    region_type : str
        rectangle, ellipse, or polygon specifying the selection widget to use.
    """

    # choose file interactively
    if file_path is None:
        file_path = Path(file_dialog(message='choose file', filter='*.txt; *.csv')[0])
    else:
        file_path = Path(file_path)

    print(file_path)

    # load data
    dat = io.load_locdata(path=file_path, file_type=file_type)

    # set roi
    rois = select_by_drawing_mpl(locdata=dat, bin_size=50, rescale='equal', region_type=region_type)
    print(rois)

    # save roi
    for i, roi in enumerate(rois):
        roi_file = file_path.stem + roi_file_indicator + f'_{i}.yaml'
        roi_path = file_path.with_name(roi_file)
        roi.to_yaml(path=roi_path)


def _add_arguments(parser):
    parser.add_argument('-f', '--file', dest='file', type=str, default=None,
                        help='File path to localization data.')
    parser.add_argument('-t', '--type', dest='type', type=int, default=1,
                        help='Integer or string indicating the file type.')
    parser.add_argument('-i', '--indicator', dest='roi_file_indicator', type=str, default='_roi',
                        help='Indicator to add to the localization file name and use as roi file name '
                             '(with further extension .yaml).')
    parser.add_argument('-r', '--region', dest='region_type', type=str, default='rectangle',
                        help='String indicating the region type.')


def main(args=None):

    parser = argparse.ArgumentParser(description='Set roi by drawing a boundary.')
    _add_arguments(parser)
    returned_args = parser.parse_args(args)

    sc_draw_roi_mpl(returned_args.file, returned_args.type, returned_args.roi_file_indicator,
                    returned_args.region_type)


if __name__ == '__main__':
    main()
