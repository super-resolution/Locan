"""

Utility functions for interacting with napari.

"""
import logging

import numpy as np

from locan.data.region import Ellipse, Polygon, Rectangle, RoiRegion
from locan.data.rois import Roi
from locan.dependencies import HAS_DEPENDENCY
from locan.visualize.napari.render2d import render_2d_napari

if HAS_DEPENDENCY["napari"]:
    import napari

logger = logging.getLogger(__name__)

__all__ = ["select_by_drawing_napari"]


def _napari_shape_to_region(vertices, bin_edges, region_type):
    """
    Convert napari shape to `locan.Region`.

    Parameters
    ----------
    vertices : numpy.ndarray of float
        Sequence of point coordinates as returned by napari.
    bin_edges : tuple, list, numpy.ndarray of float with shape (n_dimension, n_bin_edges)
        Array of bin edges for each dimension. At this point there are only equally-sized bins allowed.
    region_type : str
        String specifying the selector widget that can be either rectangle, ellipse, or polygon.

    Returns
    -------
    Region
    """
    # at this point there are only equally-sized bins used.
    bin_sizes = [bedges[1] - bedges[0] for bedges in bin_edges]

    vertices = np.array(
        [
            bedges[0] + vert * bin_size
            for vert, bedges, bin_size in zip(vertices.T, bin_edges, bin_sizes)
        ]
    ).T

    if region_type == "rectangle":
        if len(set(vertices[:, 0].astype(int))) != 2:
            raise NotImplementedError("Rotated rectangles are not implemented.")
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        corner_x, corner_y = mins
        width, height = maxs - mins
        angle = 0
        region = Rectangle((corner_x, corner_y), width, height, angle)

    elif region_type == "ellipse":
        if len(set(vertices[:, 0].astype(int))) != 2:
            raise NotImplementedError("Rotated ellipses are not implemented.")
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        width, height = maxs - mins
        center_x, center_y = mins[0] + width / 2, mins[1] + height / 2
        angle = 0
        region = Ellipse((center_x, center_y), width, height, angle)

    elif region_type == "polygon":
        region = Polygon(np.concatenate([vertices, [vertices[0]]], axis=0))

    else:
        raise TypeError(f" Type {region_type} is not defined.")

    return region


def _napari_shape_to_RoiRegion(vertices, bin_edges, region_type):
    """
    Convert napari shape to locan RoiRegion

    Parameters
    ----------
    vertices : numpy.ndarray of float
        Sequence of point coordinates as returned by napari
    bin_edges : tuple, list, numpy.ndarray of float with shape (n_dimension, n_bin_edges)
        Array of bin edges for each dimension. At this point there are only equally-sized bins allowed.
    region_type : str
        String specifying the selector widget that can be either rectangle, ellipse, or polygon.

    Returns
    -------
    RoiRegion

    Warnings
    --------
    This function is only used by :class:`locan.RoiLegacy_0` and will be deprecated.
    Use :func:`locan.render.render2d._napari_shape_to_region` instead.
    """
    # at this point there are only equally-sized bins used.
    bin_sizes = [bedges[1] - bedges[0] for bedges in bin_edges]

    # flip since napari returns vertices with first component representing the horizontal axis
    vertices = np.flip(vertices, axis=1)

    vertices = np.array(
        [
            bedges[0] + vert * bin_size
            for vert, bedges, bin_size in zip(vertices.T, bin_edges, bin_sizes)
        ]
    ).T

    if region_type == "rectangle":
        if len(set(vertices[:, 0].astype(int))) != 2:
            raise NotImplementedError("Rotated rectangles are not implemented.")
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        corner_x, corner_y = mins
        width, height = maxs - mins
        angle = 0
        region_specs = ((corner_x, corner_y), width, height, angle)

    elif region_type == "ellipse":
        if len(set(vertices[:, 0].astype(int))) != 2:
            raise NotImplementedError("Rotated ellipses are not implemented.")
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        width, height = maxs - mins
        center_x, center_y = mins[0] + width / 2, mins[1] + height / 2
        angle = 0
        region_specs = ((center_x, center_y), width, height, angle)

    elif region_type == "polygon":
        region_specs = np.concatenate([vertices, [vertices[0]]], axis=0)

    else:
        raise TypeError(f" Type {region_type} is not defined in locan.")

    return RoiRegion(region_specs=region_specs, region_type=region_type)


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
    viewer, bins = render_2d_napari(locdata, **kwargs)
    if napari_run:
        napari.run()

    vertices = viewer.layers["Shapes"].data
    types = viewer.layers["Shapes"].shape_type

    regions = []
    for verts, typ in zip(vertices, types):
        regions.append(_napari_shape_to_region(verts, bins.bin_edges, typ))

    roi_list = [Roi(reference=locdata, region=reg) for reg in regions]
    return roi_list
