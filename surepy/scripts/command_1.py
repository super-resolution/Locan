#!/usr/bin/env python

"""
Script to do something.

<Description>

To run the script::

    draw_roi -d <directory> -t <file type> -i <roi file indicator> -r <region type>

Try for instance::

    draw_roi -d "surepy/tests/test_data/five_blobs.txt" -t 1 -i "_roi" -r "ellipse"

See Also
--------
surepy.data.rois.select_by_drawing_mpl : function to draw roi

"""
from pathlib import Path


def command_1(x=None, y=1):
    """
    Description

    Parameters
    ----------
    x
    y

    Returns
    -------

    """
    print("This is command 1.")
