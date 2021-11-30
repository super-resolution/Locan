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
import re
import argparse
from pathlib import Path

from locan.constants import _has_napari
if _has_napari: import napari

from locan.constants import FileType
from locan.gui.io import file_dialog
import locan.locan_io.locdata.io_locdata as io
from locan.render.render2d import render_2d_napari


def sc_napari(file_path=None, file_type=FileType.CUSTOM, **kwargs):
    """
    Render localization data in napari.

    Parameters
    ----------
    file_path : str, os.PathLike
        File path to localization data.
    file_type : int, str, locan.constants.FileType, locan.data.metadata_pb2.Metadata
        Indicator for the file type.
        Integer or string should be according to locan.constants.FileType.
    kwargs : dict
        Other parameters passed to :func:`render_2d_napari`.
    """
    # choose file interactively
    if file_path is None:
        file_path = Path(file_dialog(message='choose file', filter=
                    'Text files (*.txt);; CSV files (*.csv);; All files (*);; All files ()'
                    )[0])
    else:
        file_path = Path(file_path)

    print(file_path)

    # load data
    dat = io.load_locdata(path=file_path, file_type=file_type)

    # render
    render_2d_napari(locdata=dat, **kwargs)
    napari.run()


def type_converter_rescale(input_string):
    if input_string == 'None':
        return None
    elif input_string == 'True':
        return True
    elif input_string == 'False':
        return False
    else:
        pattern = re.match(r'\(?([0-9]*[.]?[0-9]+),?\s?([0-9]*[.]?[0-9]+)\)?', input_string)
        if pattern:
            return tuple(float(element) for element in pattern.groups())
        else:
            return input_string


def _add_arguments(parser):
    parser.add_argument('-f', '--file', dest='file', type=str, default=None,
                        help='File path to localization data.')
    parser.add_argument('-t', '--type', dest='type', type=int, default=2,
                        help='Integer or string indicating the file type.')
    parser.add_argument('--bin_size', dest='bin_size', type=float, default=10,
                        help='The size of bins in units of locdata coordinate units. '
                             'Keyword passed to render_2d_napari function.')
    parser.add_argument('--rescale', dest='rescale', type=type_converter_rescale, default='equal',
                        help='Rescale intensity values. Keyword passed to render_2d_napari function.')


def main(args=None):

    parser = argparse.ArgumentParser(description='Render localization data in napari.')
    _add_arguments(parser)
    returned_args = parser.parse_args(args)

    sc_napari(returned_args.file, returned_args.type, bin_size=returned_args.bin_size, rescale=returned_args.rescale)


if __name__ == '__main__':
    main()
