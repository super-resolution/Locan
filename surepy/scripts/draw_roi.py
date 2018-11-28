#!/usr/bin/env python

'''
Script to define regions of interest by drawing a boundary.

With this script you can choose a file name, open the localization file and draw a rectangular region of interest.
Within the matplotlib image draw a rectange, type '+' to add the roi to the list, then type 'q' to quit.
The roi is then saved as _roi.yaml file.

To run the script::

    python draw_roi.py -d <directory> -t <file type> -i <roi file indicator>
'''

import argparse
from pathlib import Path

from surepy.gui.io import file_dialog
import surepy.io.io_locdata as io
from surepy.data.rois import Roi, select_by_drawing


def draw_roi(directory=None, type=1, roi_file_indicator='_roi'):
    """
    Define regions of interest by drawing a boundary.

    Parameters
    ----------
    directory : string or Path object
        Directory to start the GUI in for loading data.
    type : Int or str
        Integer or string indicating the file type.
        The integer should be according to surepy.data.metadata_pb2.file_type.
    roi_file_indicator : str
        Indicator to add to the localization file name and use as roi file name (with further extension .yaml).
    """

    # choose file interactively
    file = Path(file_dialog(directory=str(directory), message='choose file', filter='*.txt; *.csv')[0])
    print(file)

    # load data
    dat = io.load_locdata(path=file, type=type)

    # set roi
    rois = select_by_drawing(locdata=dat, bin_size=50, rescale='equal')
    print(rois)

    # save roi
    roi_file = file.stem + roi_file_indicator + '.yaml'
    roi_path = file.with_name(roi_file)

    rois[0].to_yaml(path=roi_path)
    # todo can we save the list of rois to one file?


def main(args=None):

    parser = argparse.ArgumentParser(description='Set roi by drawing a boundary.')

    parser.add_argument('-d', '--directory', dest='directory', type=str,
                        help='Directory to start the GUI in for loading data.')
    parser.add_argument('-t', '--type', dest='type', type=int, default=1,
                        help='Integer or string indicating the file type.')
    parser.add_argument('-i', '--indicator', dest='roi_file_indicator', type=str, default='_roi',
                        help='Indicator to add to the localization file name and use as roi file name (with further extension .yaml).')

    returned_args = parser.parse_args(args)

    draw_roi(returned_args.directory, returned_args.type, returned_args.roi_file_indicator)


if __name__ == '__main__':
    main()
