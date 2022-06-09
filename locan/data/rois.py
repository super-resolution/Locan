"""

Region of interest.

This module provides functions for managing regions of interest in localization data.

The Roi class is an object that defines a region of interest within a localization
dataset. It is therefore related to region specifications and a unique LocData object.

The Roi object provides methods for saving all specifications to a yaml file, for loading them, and for returning
LocData with localizations selected to be within the roi region.
"""
import sys
from pathlib import Path
import warnings
from itertools import product
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, PolygonSelector, EllipseSelector
from ruamel.yaml import YAML
from google.protobuf import json_format

from locan.data import metadata_pb2
from locan.data.locdata import LocData
from locan.data.metadata_utils import _modify_meta
import locan.constants
import locan.locan_io.locdata.io_locdata as io
from locan.data.region import *


__all__ = ["Roi", "rasterize"]

logger = logging.getLogger(__name__)


class _MplSelector:  # pragma: no cover
    """
    Class to use matplotlib widgets (RectangleSelector, EllipseSelector or PolygonSelector) on rendered localization
    data.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to use the widget on.
    type : str
        Type is a string specifying the selector widget that can be either rectangle, ellipse, or polygon.

    Attributes
    ----------
    rois : List of dict
        A list of rois where each element is a dict with keys `region_specs` and 'region'.
        `region_specs` contain a tuple with specifications for the chosen region type (see ``Roi``).
        The region is a string identifier that can be either rectangle, ellipse, or polygon.
    """

    def __init__(self, ax, type="rectangle"):
        self.rois = []
        self.ax = ax

        if type == "rectangle":
            self.selector = RectangleSelector(
                self.ax, self.selector_callback, drawtype="box", interactive=True
            )
            self.type = type

        elif type == "ellipse":
            self.selector = EllipseSelector(self.ax, self.selector_callback)
            self.type = type

        elif type == "polygon":
            raise NotImplementedError(
                "The polygon selection is not working correctly. Use napari."
            )
        # The PolygonSelector is not working correctly.
        #     self.selector = PolygonSelector(self.ax, self.p_selector_callback)
        #     self.type = type

        else:
            raise TypeError("Type {} is not defined.".format(type))

        plt.connect("key_press_event", self.key_pressed_callback)

    def selector_callback(self, eclick, erelease):
        """ eclick and erelease are matplotlib events at press and release."""
        print("startposition: {}".format((eclick.xdata, eclick.ydata)))
        print("endposition  : {}".format((erelease.xdata, erelease.ydata)))

    def p_selector_callback(self, vertices):
        print("Vertices: {}".format(vertices))

    def key_pressed_callback(self, event):
        print("Key pressed.")
        if event.key in ["R", "r"]:
            print("RectangleSelector activated.")
            self.selector = RectangleSelector(
                self.ax, self.selector_callback, drawtype="box", interactive=True
            )
            self.type = "rectangle"

        elif event.key in ["E", "e"]:
            print("EllipseSelector activated.")
            self.selector = EllipseSelector(
                self.ax, self.selector_callback, drawtype="box", interactive=True
            )
            self.type = "ellipse"

        elif event.key in ["T", "t"]:
            print("PolygonSelector activated.")
            self.selector = PolygonSelector(self.ax, self.p_selector_callback)
            self.type = "polygon"

        else:
            pass

        if event.key in ["+"] and self.selector.active and self.type == "rectangle":
            print("Roi was added.")
            region_specs = (
                np.flip(self.selector.geometry[:, 0]),
                self.selector.extents[1] - self.selector.extents[0],
                self.selector.extents[3] - self.selector.extents[2],
                0.0,
            )
            self.rois.append({"region_specs": region_specs, "region": self.type})
            print("rois: {}".format(self.rois))

        elif event.key in ["+"] and self.selector.active and self.type == "ellipse":
            print("Roi was added.")
            region_specs = (
                self.selector.center,
                self.selector.extents[1] - self.selector.extents[0],
                self.selector.extents[3] - self.selector.extents[2],
                0.0,
            )

            self.rois.append({"region_specs": region_specs, "region": self.type})
            print("rois: {}".format(self.rois))

        elif event.key in ["+"] and self.selector.active and self.type == "polygon":
            print("Roi was added.")
            vertices_ = self.selector.verts.append(self.selector.verts[0])
            self.rois.append({"region_specs": vertices_, "region": self.type})
            print("rois: {}".format(self.rois))

        else:
            pass


class Roi:
    """
    Class for a region of interest on LocData (roi).

    Roi objects define a region of interest for a referenced LocData object.

    Parameters
    ----------
    region : Region
        Geometrical region of interest.
    reference : LocData, dict, locan.data.metadata_pb2.Metadata, None
        Reference to localization data for which the region of interests are defined. It can be a LocData object,
        a reference to a saved SMLM file, or None for indicating no specific reference.
        When referencing a saved SMLM file, reference must be a dict or locan.data.metadata_pb2.Metadata with keys
        `file.path` and `file.type` for a path pointing to a localization file and an integer or string
        indicating the file type.
        Integer or string should be according to locan.constants.FileType.
    loc_properties : tuple of str
        Localization properties in LocData object on which the region selection will be applied (for instance the
        coordinate_labels).


    Attributes
    ----------
    region : Region
        Geometrical region of interest.
    reference : LocData, locan.data.metadata_pb2.Metadata, None
        Reference to localization data for which the region of interests are defined. It can be a LocData object,
        a reference (locan.data.metadata_pb2.Metadata) to a saved SMLM file, or None for indicating no specific reference.
        When referencing a saved SMLM file, reference as attributes `file.path`
        and `file.type` for a path pointing to a localization file and an integer indicating the file type.
        The integer should be according to locan.data.metadata_pb2.Metadata.file.type.
    loc_properties : tuple of str
        Localization properties in LocData object on which the region selection will be applied (for instance the
        coordinate_labels).
    """

    def __init__(self, region, reference=None, loc_properties=()):
        if isinstance(reference, dict):
            self.reference = metadata_pb2.Metadata()
            self.reference.file.path = str(reference["file_path"])
            ft_ = reference["file_type"]

            if isinstance(reference["file_type"], int):
                self.reference.file.type = ft_
            elif isinstance(reference["file_type"], str):
                self.reference.file.type = locan.constants.FileType[ft_.upper()].value
            elif isinstance(reference["file_type"], locan.constants.FileType):
                self.reference.file.type = ft_
            elif isinstance(reference["file_type"], metadata_pb2):
                self.reference.file.type = ft_.file.type
            else:
                raise TypeError
        elif isinstance(reference, metadata_pb2.Metadata):
            self.reference = metadata_pb2.Metadata()
            self.reference.MergeFrom(reference)
        else:
            self.reference = reference

        self.region = region
        self.loc_properties = loc_properties

    def __repr__(self):
        return (
            f"Roi(reference={self.reference}, "
            f"region={repr(self.region)}, "
            f" loc_properties={self.loc_properties})"
        )

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, region_):
        if not isinstance(region_, Region):
            raise TypeError("An instance of locan.Region must be provided.")
        self._region = region_

    def to_yaml(self, path=None):
        """
        Save Roi object in yaml format.

        Parameters
        ----------
        path : str, bytes, os.PathLike, None
            Path for yaml file. If None a roi file path is generated from the metadata.
        """

        # prepare path
        if path is None and isinstance(self.reference, LocData):
            _file_path = Path(self.reference.meta.file.path)
            _roi_file = _file_path.stem + "_roi.yaml"
            _path = _file_path.with_name(_roi_file)
        elif path is None and isinstance(self.reference, metadata_pb2.Metadata):
            _file_path = Path(self.reference.file.path)
            _roi_file = _file_path.stem + "_roi.yaml"
            _path = _file_path.with_name(_roi_file)
        else:
            _path = Path(path)

        # prepare reference for yaml representation - reference to LocData cannot be represented
        if self.reference is None:
            reference_for_yaml = None
        elif isinstance(self.reference, LocData):
            if self.reference.meta.file.path:
                meta_ = metadata_pb2.Metadata()
                meta_.file.path = self.reference.meta.file.path
                meta_.file.type = self.reference.meta.file.type
                reference_for_yaml = json_format.MessageToJson(
                    meta_, including_default_value_fields=False
                )
            else:
                warnings.warn(
                    "The localization data has to be saved and the file path provided, "
                    "or the reference is lost.",
                    UserWarning,
                )
                reference_for_yaml = None
        else:
            reference_for_yaml = json_format.MessageToJson(
                self.reference, including_default_value_fields=False
            )

        region_for_yaml = repr(self.region)
        loc_properties_for_yaml = self.loc_properties

        yaml = YAML()
        output = dict(
            reference=reference_for_yaml,
            region=region_for_yaml,
            loc_properties=loc_properties_for_yaml,
        )
        yaml.dump(output, _path)

    @classmethod
    def from_yaml(cls, path):
        """
        Read Roi object from yaml format.

        Parameters
        ----------
        path : str, bytes, os.PathLike
            Path for yaml file.
        """
        yaml = YAML(typ="safe")
        with open(path) as file:
            yaml_output = yaml.load(file)

        if yaml_output["reference"] is not None:
            reference_ = metadata_pb2.Metadata()
            reference_ = json_format.Parse(yaml_output["reference"], reference_)
        else:
            reference_ = yaml_output["reference"]

        region_ = eval(yaml_output["region"])
        loc_properties_ = yaml_output["loc_properties"]

        return cls(reference=reference_, region=region_, loc_properties=loc_properties_)

    def locdata(self, reduce=True):
        """
        Localization data according to roi specifications.

        The ROI is applied on locdata properties as specified in self.loc_properties or by taking the first
        applicable locdata.coordinate_labels.

        Parameters
        ----------
        reduce : Bool
            Return the reduced LocData object or keep references alive.

        Returns
        -------
        LocData
            A new instance of LocData with all localizations within region of interest.
        """
        local_parameter = locals()

        if isinstance(self.reference, LocData):
            locdata = self.reference
        elif isinstance(self.reference, metadata_pb2.Metadata):
            locdata = io.load_locdata(
                self.reference.file.path, self.reference.file.type
            )
        else:
            raise AttributeError("Valid reference to locdata is missing.")

        if not len(locdata):
            logger.warning("Locdata in reference is empty.")
            new_locdata = locdata

        else:
            if self.loc_properties:
                pfr = self.loc_properties
            else:
                pfr = locdata.coordinate_labels[0 : self.region.dimension]

            points = locdata.data[list(pfr)].values
            indices_inside = self._region.contains(points)

            locdata_indices_to_keep = locdata.data.index[indices_inside]
            new_locdata = LocData.from_selection(
                locdata=locdata, indices=locdata_indices_to_keep
            )
            if pfr == new_locdata.coordinate_labels:
                new_locdata.region = self._region
            else:
                new_locdata.region = None

        # finish
        if reduce:
            new_locdata.reduce()

        # update metadata
        meta_ = _modify_meta(
            locdata,
            new_locdata,
            function_name=sys._getframe().f_code.co_name,
            parameter=local_parameter,
            meta=None,
        )
        new_locdata.meta = meta_

        return new_locdata


class RoiLegacy_0:
    """
    Class for a region of interest on LocData (roi).

    Roi objects define a region of interest for a referenced LocData object.

    Parameters
    ----------
    reference : LocData, dict, locan.data.metadata_pb2.Metadata, None
        Reference to localization data for which the region of interests are defined. It can be a LocData object,
        a reference to a saved SMLM file, or None for indicating no specific reference.
        When referencing a saved SMLM file, reference must be a dict or locan.data.metadata_pb2.Metadata with keys `file_path`
        and `file_type` for a path pointing to a localization file and an integer or string indicating the file type.
        Integer or string should be according to locan.constants.FileType.
    region_type : str
        A string indicating the roi shape.
        In 1D it can be `interval`.
        In 2D it can be either `rectangle`, `ellipse`, or closed `polygon`.
        In 3D it can be either `cuboid` or `ellipsoid` or `polyhedron` (not implemented yet).
    region_specs : tuple
        1D rois are defined by the following tuple:
        * interval: (start, stop)
        2D rois are defined by the following tuples:
        * rectangle: ((corner_x, corner_y), width, height, angle)
        * ellipse: ((center_x, center_y), width, height, angle)
        * polygon: ((point1_x, point1_y), (point2_x, point2_y), ..., (point1_x, point1_y))
        3D rois are defined by the following tuples:
        * cuboid: ((corner_x, corner_y, corner_z), length, width, height, angle_1, angle_2, angle_3)
        * ellipsoid: ((center_x, center_y, center_z), length, width, height, angle_1, angle_2, angle_3)
        * polyhedron: (...)
    properties_for_roi : tuple of str
        Localization properties in LocData object on which the region selection will be applied (for instance the
        coordinate_labels).


    Attributes
    ----------
    reference : LocData, locan.data.metadata_pb2.Metadata, None
        Reference to localization data for which the region of interests are defined. It can be a LocData object,
        a reference (locan.data.metadata_pb2.Metadata) to a saved SMLM file, or None for indicating no specific reference.
        When referencing a saved SMLM file, reference as attributes `file_path`
        and `file_type` for a path pointing to a localization file and an integer indicating the file type.
        The integer should be according to locan.data.metadata_pb2.Metadata.file_type.
    _region : RoiRegion, list of RoiRegion
        Object specifying the geometrical region of interest. In case a list of RoiRegion is provided it is the union
        that makes up the region of interest.
    properties_for_roi : tuple of str
        Localization properties in LocData object on which the region selection will be applied (for instance the
        coordinate_labels).

    Warnings
    --------
    `RoiLegacy` is deprecated and should only be used to read legacy _roi.yaml files. Use :class:`locan.Roi` instead.
    """

    def __init__(
        self, region_type, region_specs, reference=None, properties_for_roi=()
    ):
        if isinstance(reference, dict):
            self.reference = metadata_pb2.Metadata()
            self.reference.file.path = str(reference["file_path"])
            ft_ = reference["file_type"]

            if isinstance(reference["file_type"], int):
                self.reference.file.type = ft_
            elif isinstance(reference["file_type"], str):
                self.reference.file.type = locan.constants.FileType[ft_.upper()].value
            elif isinstance(reference["file_type"], locan.constants.FileType):
                self.reference.file.type = ft_
            elif isinstance(reference["file_type"], metadata_pb2):
                self.reference.file.type = ft_.file.type
            else:
                raise TypeError

        elif isinstance(reference, metadata_pb2.Metadata):
            self.reference = metadata_pb2.Metadata()
            self.reference.MergeFrom(reference)
        else:
            self.reference = reference

        self._region = RoiRegion(region_type=region_type, region_specs=region_specs)
        self.properties_for_roi = properties_for_roi

    def __repr__(self):
        return (
            f"Roi(reference={self.reference}, "
            f"region={self._region.region_type}, "
            f"region_specs={self._region.region_specs},"
            f" loc_properties={self.properties_for_roi})"
        )

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, region_):
        if isinstance(region_, RoiRegion):
            self._region = region_

    def to_yaml(self, path=None):
        """
        Save Roi object in yaml format.

        Parameters
        ----------
        path : str, bytes, os.PathLike, None
            Path for yaml file. If None a roi file path is generated from the metadata.
        """
        warnings.warn(
            "RoiLegacy.to_yaml is deprecated, use Roi.to_yaml instead",
            DeprecationWarning,
        )

        # prepare path
        if path is None and isinstance(self.reference, LocData):
            _file_path = Path(self.reference.meta.file.path)
            _roi_file = _file_path.stem + "_roi.yaml"
            _path = _file_path.with_name(_roi_file)
        elif path is None and isinstance(self.reference, metadata_pb2.Metadata):
            _file_path = Path(self.reference.file.path)
            _roi_file = _file_path.stem + "_roi.yaml"
            _path = _file_path.with_name(_roi_file)
        else:
            _path = Path(path)

        # prepare reference for yaml representation - reference to LocData cannot be represented
        if self.reference is None:
            reference_for_yaml = None
        elif isinstance(self.reference, LocData):
            if self.reference.meta.file.path:
                meta_ = metadata_pb2.Metadata()
                meta_.file.path = self.reference.meta.file.path
                meta_.file.type = self.reference.meta.file.type
                reference_for_yaml = json_format.MessageToJson(
                    meta_, including_default_value_fields=False
                )
            else:
                warnings.warn(
                    "The localization data has to be saved and the file path provided, "
                    "or the reference is lost.",
                    UserWarning,
                )
                reference_for_yaml = None
        else:
            reference_for_yaml = json_format.MessageToJson(
                self.reference, including_default_value_fields=False
            )

        # prepare points for yaml representation - numpy.float has to be converted to float
        def nested_change(iterable, func):
            if isinstance(iterable, (list, tuple, np.ndarray)):
                return [nested_change(x, func) for x in iterable]
            return func(iterable)

        region_specs_for_yaml = nested_change(self._region.region_specs, float)

        region_type_for_yaml = self._region.region_type
        properties_for_roi_for_yaml = self.properties_for_roi

        yaml = YAML()
        output = dict(
            reference=reference_for_yaml,
            region_type=region_type_for_yaml,
            region_specs=region_specs_for_yaml,
            properties_for_roi=properties_for_roi_for_yaml,
        )
        yaml.dump(output, _path)

    @classmethod
    def from_yaml(cls, path):
        """
        Read Roi object from yaml format.

        Parameters
        ----------
        path : str, bytes, os.PathLike
            Path for yaml file.
        """
        yaml = YAML(typ="safe")
        with open(path) as file:
            yaml_output = yaml.load(file)

        if yaml_output["reference"] is not None:
            reference_ = metadata_pb2.Metadata()
            reference_ = json_format.Parse(yaml_output["reference"], reference_)
        else:
            reference_ = yaml_output["reference"]

        region_type_ = yaml_output["region_type"]
        region_specs_ = yaml_output["region_specs"]
        properties_for_roi_ = yaml_output["properties_for_roi"]

        return cls(
            reference=reference_,
            region_type=region_type_,
            region_specs=region_specs_,
            properties_for_roi=properties_for_roi_,
        )

    def locdata(self, reduce=True):
        """
        Localization data according to roi specifications.

        The ROI is applied on locdata properties as specified in self.loc_properties or by taking the first
        applicable locdata.coordinate_labels.

        Parameters
        ----------
        reduce : Bool
            Return the reduced LocData object or keep references alive.

        Returns
        -------
        LocData
            A new instance of LocData with all localizations within region of interest.
        """
        local_parameter = locals()

        if isinstance(self.reference, LocData):
            locdata = self.reference
        elif isinstance(self.reference, metadata_pb2.Metadata):
            locdata = io.load_locdata(
                self.reference.file_path, self.reference.file_type
            )
        else:
            raise AttributeError("Valid reference to locdata is missing.")

        if not len(locdata):
            logger.warning("Locdata in reference is empty.")
            new_locdata = locdata

        else:
            if self.properties_for_roi:
                pfr = self.properties_for_roi
            else:
                pfr = locdata.coordinate_labels[0 : self._region.dimension]

            points = locdata.data[list(pfr)].values
            indices_inside = self._region.contains(points)

            locdata_indices_to_keep = locdata.data.index[indices_inside]
            new_locdata = LocData.from_selection(
                locdata=locdata, indices=locdata_indices_to_keep
            )
            if pfr == new_locdata.coordinate_labels:
                new_locdata.region = self._region
            else:
                new_locdata.region = None

        # finish
        if reduce:
            new_locdata.reduce()

        # update metadata
        meta_ = _modify_meta(
            locdata,
            new_locdata,
            function_name=sys._getframe().f_code.co_name,
            parameter=local_parameter,
            meta=None,
        )
        new_locdata.meta = meta_

        return new_locdata


# todo generalize to take all loc_properties
def rasterize(locdata, support=None, n_regions=(2, 2, 2), loc_properties=()):
    """
    Provide regions of interest by dividing the locdata support in equally sized rectangles.

    Parameters
    ----------
    locdata : LocData
        The localization data from which to select localization data.
    support : tuple of tuples, None
        Coordinate intervals that are divided in `n_regions` subintervals.
        For None intervals are taken from the bounding box.
    n_regions : tuple with size 2 or 3.
        Number of regions in each dimension. E.g. `n_regions` = (2, 2) returns 4 rectangular Roi objects.
    loc_properties : tuple of str
        Localization properties in LocData object on which the region selection will be applied.
        (Only implemented for coordinates labels)

    Returns
    -------
    tuple(locan.data.rois.Roi)
        A sequence of Roi objects
    """
    if len(locdata) == 0:
        raise ValueError("Not implemented for empty LocData objects.")

    if not set(loc_properties).issubset(locdata.coordinate_labels):
        raise ValueError("loc_properties must be tuple with coordinate labels.")

    if loc_properties:
        coordinate_labels_indices = [
            locdata.coordinate_labels.index(pfr) for pfr in loc_properties
        ]
    else:
        coordinate_labels_indices = range(len(n_regions))

    # specify support
    if support is None:
        support_ = locdata.bounding_box.vertices[coordinate_labels_indices]
        if len(locdata.bounding_box.width) == 0:
            widths = np.zeros(len(n_regions))
        else:
            widths = locdata.bounding_box.width[coordinate_labels_indices] / n_regions
    else:
        support_ = support
        widths = np.diff(support_).flatten() / n_regions

    # specify interval corners
    corners = [
        np.linspace(*support_d, n_regions_d, endpoint=False)
        for support_d, n_regions_d in zip(support_, n_regions)
    ]
    corners = product(*corners)

    # specify regions
    if len(n_regions) == 2:
        regions = [Rectangle(corner, *widths, 0) for corner in corners]

    elif len(n_regions) == 3:
        raise NotImplementedError("Computation for 3D has not been implemented, yet.")

    else:
        raise ValueError("The shape of n_regions is incompatible.")

    new_rois = tuple(
        [
            Roi(reference=locdata, region=reg, loc_properties=loc_properties)
            for reg in regions
        ]
    )
    return new_rois
