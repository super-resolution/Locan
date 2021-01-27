#!/usr/bin/env python

"""
Render localization data in napari.

With this script you can choose a file name and render the localization file in napari.

To run the script::

    napari -f <file> -t <file type> -k <string with kwrds for render function>

Try for instance::

    napari -f "surepy/tests/test_data/five_blobs.txt" -t 1 -k "bin_size=50"

See Also
--------
surepy.render.render2d.render_2d_napari
"""
import argparse
from pathlib import Path

from surepy.constants import _has_napari
if _has_napari: import napari

from surepy.constants import FileType
from surepy.gui.io import file_dialog
import surepy.io.io_locdata as io
from surepy.render.render2d import render_2d_napari


def sc_napari(file_path=None, file_type=FileType.CUSTOM, **kwargs):
    """
    Render localization data in napari.

    Parameters
    ----------
    file_path : str, os.PathLike
        File path to localization data.
    file_type : int, str, surepy.constants.FileType, surepy.data.metadata_pb2.Metadata
        Indicator for the file type.
        Integer or string should be according to surepy.constants.FileType.

    Other Parameters
    ----------------
    kwargs : dict
        Keywords passed to :func:`render_2d_napari`.
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
