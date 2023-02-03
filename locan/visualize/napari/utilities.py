"""

Utility functions for interacting with napari.

"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable  # noqa: F401

import numpy as np

from locan.data.region import Ellipse, Polygon, Rectangle, Region
from locan.data.rois import Roi
from locan.dependencies import HAS_DEPENDENCY
from locan.gui import file_dialog
from locan.visualize.napari.render2d import render_2d_napari

if HAS_DEPENDENCY["napari"]:
    import napari

logger = logging.getLogger(__name__)

__all__ = [
    "select_by_drawing_napari",
    "get_rois",
    "save_rois",
]


def select_by_drawing_napari(locdata, napari_run=True, **kwargs):
    """
    Select region of interest from rendered image by drawing rois in napari.

    Rois will be created from shapes in napari.viewer.layers['Shapes'].

    Parameters
    ----------
    locdata : LocData
        The localization data from which to select localization data.
    napari_run : bool
        If `True` napari.run is called (set to `False` for testing).
    kwargs : dict
        Other parameters passed to :func:`render_2d_napari`.

    Returns
    -------
    list[Roi]

    See Also
    --------
    :func:`locan.scripts.rois` : script for drawing rois
    """
    # select roi
    viewer = render_2d_napari(locdata, **kwargs)
    if "Rois" not in viewer.layers:
        viewer.add_shapes(name="Rois", edge_width=0.1)
    if napari_run:
        napari.run()

    roi_list = get_rois(viewer.layers["Rois"], reference=locdata)

    return roi_list


def _shape_to_region(vertices, shape_type):
    """
    Convert napari shape to `locan.Region`.

    Parameters
    ----------
    vertices : np.array[float]
        Sequence of point coordinates as returned by napari.
    shape_type : str
        One of rectangle, ellipse, or polygon.

    Returns
    -------
    Region
    """
    if shape_type == "rectangle":
        if len(set(vertices[:, 0].astype(int))) != 2:
            raise NotImplementedError("Rotated rectangles are not implemented.")
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        corner_x, corner_y = mins
        width, height = maxs - mins
        angle = 0
        region = Rectangle((corner_x, corner_y), width, height, angle)

    elif shape_type == "ellipse":
        if len(set(vertices[:, 0].astype(int))) != 2:
            raise NotImplementedError("Rotated ellipses are not implemented.")
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        width, height = maxs - mins
        center_x, center_y = mins[0] + width / 2, mins[1] + height / 2
        angle = 0
        region = Ellipse((center_x, center_y), width, height, angle)

    elif shape_type == "polygon":
        region = Polygon(np.concatenate([vertices, [vertices[0]]], axis=0))

    else:
        raise TypeError(f" Type {shape_type} is not defined.")

    return region


def _shapes_to_regions(shapes_data) -> list[Region]:
    """
    Convert napari shapes to `locan.Region`.

    Parameters
    ----------
    shapes_data : napari.types.ShapesData
        Shapes data with list of shapes

    Returns
    -------
    list[Region]
    """
    if shapes_data[2] != "shapes":
        raise ValueError("shapes_data[2] must equal 'shapes'.")
    data = shapes_data[0]
    shape_types = shapes_data[1]["shape_type"]
    regions = [
        _shape_to_region(vertices=vertices, shape_type=shape_type)
        for vertices, shape_type in zip(data, shape_types)
    ]
    return regions


def get_rois(shapes_layer, reference=None, loc_properties=None):
    """
    Create rois from shapes in napari.viewer.Shapes.

    Parameters
    ----------
    shapes_layer: napari.layers.Shapes
        Napari shapes layer like `viewer.layers["Shapes"]`
    reference : LocData, dict, locan.data.metadata_pb2.Metadata, locan.data.metadata_pb2.File, None
        Reference to localization data for which the region of interest
        is defined. It can be a LocData object, a reference to a saved
        SMLM file, or None for indicating no specific reference.
        When dict it must have keys `file_path`and `file_type`.
        When Metadata message it must have keys `file.path` and
        `file.type` for a path pointing to a localization file and an
        integer or string indicating the file type.
        Integer or string should be according to
        locan.constants.FileType.
    loc_properties : tuple of str
        Localization properties in LocData object on which the region
        selection will be applied (for instance the coordinate_labels).

    Returns
    -------
    list[Roi]

    See Also
    --------
    :func:`locan.scripts.rois` : script for drawing rois
    """
    shapes_data = shapes_layer.as_layer_data_tuple()
    regions = _shapes_to_regions(shapes_data=shapes_data)
    rois = [
        Roi(region=reg, reference=reference, loc_properties=loc_properties)
        for reg in regions
    ]
    return rois


def save_rois(rois, file_path=None, roi_file_indicator="_roi"):
    """
    Save list of Roi objects.

    Parameters
    ----------
    rois : Iterable[Rois]
        The rois to be saved.
    file_path : str | bytes | os.PathLike | "roi_reference" | None
        Base name for roi files or existing directory to save rois in.
        If "roi_reference", roi.reference.file.path is used.
        If None, a file dialog is opened.
    roi_file_indicator : str
        Indicator to add to the localization file name and use as roi
        file name (with further extension .yaml).

    Returns
    -------
    list[str]
        New created roi file paths.
    """
    # choose file interactively
    if file_path is None:
        file_path = Path(
            file_dialog(message="choose localization file or base name")[0]
        )
    elif file_path == "roi_reference":
        file_path = None
    elif Path(file_path).is_dir():
        file_path = Path(file_path) / "my"
    else:
        file_path = Path(file_path)

    # create roi file names and save rois
    roi_path_list = []
    for i, roi in enumerate(rois):
        if file_path is None:
            try:
                file_path = Path(roi.reference.file.path)
            except AttributeError:
                raise
        roi_file = file_path.stem + roi_file_indicator + f"_{i}.yaml"
        roi_path = file_path.with_name(roi_file)
        roi_path_list.append(roi_path)
        roi.to_yaml(path=roi_path)

    return roi_path_list
