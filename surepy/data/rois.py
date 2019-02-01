"""

Region of interest.

This module provides functions for managing regions of interest in localization data.

The Roi class is an object that defines a region of interest within a localization
dataset. It is therefore related to region specifications and a unique LocData object.

The Roi object provides methods for saving all specifications to a yaml file, for loading them, and for returning
LocData with localizations selected to be within the roi region.

The roi region is provided as a RoiRegion object that provides methods for getting a
printable representation (that can be saved in a yaml file), for returning a matplotlib patch that can be
shown in a graph, and for finding points within the region.
"""

from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mPath
import matplotlib.patches as mPatches
from matplotlib.widgets import RectangleSelector, PolygonSelector, EllipseSelector
from scipy.spatial.distance import pdist
from ruamel.yaml import YAML
from shapely.geometry import Polygon as shPolygon
from google.protobuf import json_format

from surepy.data import metadata_pb2
from surepy import LocData
import surepy.io.io_locdata as io
from surepy.render import render2D


class RoiRegion:
    """
    Region object to specify regions of interest.

    A region that defines a region of interest with methods for getting a printable representation (that can also be
    saved in a yaml file), for returning a matplotlib patch that can be shown in a graph, for finding points within
    the region.

    Parameters
    ----------
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


    Attributes
    ----------
    region_type : str
        Type of region
    region_specs : tuple
        Specifications for region
    _region : RoiRegion object
        RoiRegion instance for the specified region type.
    polygon : ndarray of tuples
        Array of points for a closed polygon approximating the region of interest in clockwise orientation. The first
        and last point must be identical.
    dimension : int
        Spatial dimension of region
    centroid : tuple of float
        Centroid coordinates
    max_distance : array-like of float
        Maximum distance between any two points in the region
    region_measure : float
        Hull measure, i.e. area or volume
    subregion_measure : float
        Measure of the sub-dimensional region, i.e. circumference or surface.
    """

    def __init__(self, region_type, region_specs):
        self.region_specs = region_specs
        self.region_type = region_type

        if region_type == 'interval':
            self._region = _RoiInterval(region_specs)

        elif region_type == 'rectangle':
            self._region = _RoiRectangle(region_specs)

        elif region_type == 'ellipse':
            self._region = _RoiEllipse(region_specs)

        elif region_type == 'polygon':
            self._region = _RoiPolygon(region_specs)

        else:
            raise NotImplementedError(f'Region_type {region_specs} has not been implemented yet.')

        self.polygon = self._region.polygon
        self.dimension = self._region.dimension
        self.centroid = self._region.centroid
        self.max_distance = self._region.max_distance
        self.region_measure = self._region.region_measure
        self.subregion_measure = self._region.subregion_measure

    def __repr__(self):
        """
        Readable, printable and savable representation of RoiRegion.
        """
        return str(dict(region_type=self.region_type, region_specs=self.region_specs))

    def contains(self, points):
        """
        Return list of indices for all points that are inside the region of interest.

        Parameters
        ----------
        points : array of 2D or 3D coordinates
            Points that are tested for being inside the specified region.

        Returns
        -------
        ndarray of ints
            Array with indices for all points in original point array that are within the region.
        """
        return self._region.contains(points)

    def as_artist(self, origin=(0, 0), **kwargs):
        """
        Matplotlib patch object for this region (`matplotlib.patches.Ellipse`).

        Parameters:
        -----------
        origin : array_like, optional
            The ``(x, y)`` pixel position of the origin of the displayed image.
            Default is (0, 0).

        Other Parameters
        ----------------
        kwargs : `dict`
            Other parameters passed to the `matplotlib.patches` object.

        Returns
        -------
        patch : matplotlib.patches object
            Matplotlib patch for the specified region
        """
        return self._region.as_artist(origin=origin, **kwargs)


class _RoiInterval:

    def __init__(self, region_specs):
        if len(region_specs) != 2:
            raise AttributeError('The shape of region_specs must be (2,).')
        else:
            self.region_specs = region_specs

    def __repr__(self):
        return dict(region_type='interval', region_specs=self.region_specs)

    def contains(self, points):
        inside_indices = np.where((points >= self.region_specs[0]) & (points <= self.region_specs[1]))
        return inside_indices

    @property
    def polygon(self):
        return np.array(self.region_specs)

    @property
    def dimension(self):
        return 1

    @property
    def centroid(self):
        minimum, maximum = self.region_specs
        return minimum + (maximum - minimum)/2

    @property
    def max_distance(self):
        minimum, maximum = self.region_specs
        return maximum - minimum

    @property
    def region_measure(self):
        minimum, maximum = self.region_specs
        return maximum - minimum

    @property
    def subregion_measure(self):
        return None

    def as_artist(self, origin=(0, 0), **kwargs):
        raise NotImplementedError


class _RoiRectangle:

    def __init__(self, region_specs):
        if len(region_specs) != 4:
            raise AttributeError('The shape of region_specs must be (2,).')
        else:
            self.region_specs = region_specs

    def __repr__(self):
        return dict(region_type='rectangle', region_specs=self.region_specs)

    def contains(self, points):
        corner, width, height, angle = self.region_specs
        fig, ax = plt.subplots()
        rect = mPatches.Rectangle(corner, width, height, angle=angle, fill=False, edgecolor='b', linewidth=1)
        ax.add_patch(rect)
        polygon = rect.get_patch_transform().transform(rect.get_path().vertices)

        polygon_path = mPath.Path(polygon, closed=True)
        mask = polygon_path.contains_points(points)
        inside_indices = np.where(mask)[0]
        return inside_indices

    @property
    def polygon(self):
        corner, width, height, angle = self.region_specs
        fig, ax = plt.subplots()
        rect = mPatches.Rectangle(corner, width, height, angle=angle, fill=False, edgecolor='b', linewidth=1)
        ax.add_patch(rect)
        polygon = rect.get_patch_transform().transform(rect.get_path().vertices)
        return polygon[::-1]

    @property
    def dimension(self):
        return 2

    @property
    def centroid(self):
        corner, width, height, angle = self.region_specs
        center_before_rotation_x, center_before_rotation_y = corner[0] + width / 2, corner[1] + height / 2

        center_x = center_before_rotation_x * np.cos(angle) - center_before_rotation_y * np.sin(angle)
        center_y = center_before_rotation_x * np.sin(angle) + center_before_rotation_y * np.cos(angle)

        return center_x, center_y

    @property
    def max_distance(self):
        _, width, height, _ = self.region_specs
        return np.sqrt(width**2 + height**2)

    @property
    def region_measure(self):
        _, width, height, _ = self.region_specs
        return width * height

    @property
    def subregion_measure(self):
        _, width, height, _ = self.region_specs
        return 2 * width + 2 * height

    def as_artist(self, origin=(0, 0), **kwargs):
        from matplotlib.patches import Rectangle
        corner, width, height, angle = self.region_specs
        xy = corner[0] - origin[0], corner[1] - origin[1]
        mpl_params = self.mpl_properties_default('patch')
        mpl_params.update(kwargs)

        return Rectangle(xy=xy, width=width, height=height, angle=angle, **mpl_params)


class _RoiEllipse:

    def __init__(self, region_specs):
        if len(region_specs) != 4:
            raise AttributeError('The shape of region_specs must be (2,).')
        else:
            self.region_specs = region_specs

    def __repr__(self):
        return dict(region_type='ellipse', region_specs=self.region_specs)

    def contains(self, points):
        center, width, height, angle = self.region_specs

        cos_angle = np.cos(np.radians(-angle))
        sin_angle = np.sin(np.radians(-angle))

        xc = points[:,0] - center[0]
        yc = points[:,1] - center[1]

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle

        rad_cc = (xct**2/(width/2.)**2) + (yct**2/(height/2.)**2)

        inside_indices = np.where(rad_cc < 1.)[0]
        return inside_indices

    @property
    def polygon(self):
        center, width, height, angle = self.region_specs
        fig, ax = plt.subplots()
        rect = mPatches.Ellipse(center, width, height, angle=angle, fill=False, edgecolor='b', linewidth=1)
        ax.add_patch(rect)
        polygon = rect.get_patch_transform().transform(rect.get_path().vertices)
        return polygon[::-1]

    @property
    def dimension(self):
        return 2

    @property
    def centroid(self):
        center, _, _, _ = self.region_specs
        return center

    @property
    def max_distance(self):
        _, width, height, _ = self.region_specs
        return np.max([width, height])

    @property
    def region_measure(self):
        _, width, height, _ = self.region_specs
        return np.pi * width/2 * height/2

    @property
    def subregion_measure(self):
        _, width, height, _ = self.region_specs
        # using Ramanujan approximation
        a, b = width / 2, height / 2
        t = ((a - b) / (a + b)) ** 2
        circumference = np.pi * (a + b) * (1 + 3 * t / (10 + np.sqrt(4 - 3 * t)))
        return circumference

    def as_artist(self, origin=(0, 0), **kwargs):
        from matplotlib.patches import Ellipse
        center, width, height, angle = self.region_specs
        xy = center[0] - origin[0], center[1] - origin[1]
        mpl_params = self.mpl_properties_default('patch')
        mpl_params.update(kwargs)

        return Ellipse(xy=xy, width=width, height=height, angle=angle, **mpl_params)


class _RoiPolygon:

    def __init__(self, region_specs):
        if region_specs[0] != region_specs[-1]:
            raise ValueError('First and last element of polygon must be identical.')
        else:
            self.region_specs = region_specs

    def __repr__(self):
        return dict(region_type='polygon', region_specs=self.region_specs)

    def contains(self, points):
        polygon_path = mPath.Path(self.region_specs, closed=True)
        mask = polygon_path.contains_points(points)
        inside_indices = np.where(mask)[0]
        return inside_indices

    @property
    def polygon(self):
        return np.array(self.region_specs)

    @property
    def dimension(self):
        return 2

    @property
    def centroid(self):
        polygon = shPolygon(self.region_specs)
        return polygon.centroid.coords

    @property
    def max_distance(self):
        distances = pdist(self.region_specs[0:-1])
        return np.nanmax(distances)

    @property
    def region_measure(self):
        polygon = shPolygon(self.region_specs)
        return polygon.area

    @property
    def subregion_measure(self):
        polygon = shPolygon(self.region_specs)
        return polygon.length

    def as_artist(self, origin=(0, 0), **kwargs):
        raise NotImplementedError


class _MplSelector:
    """
    Class to use matplotlib widgets (RectangleSelector, EllipseSelector or PolygonSelector) on rendered localization
    data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to use the widget on.
    type : str
        Type is a string specifying the selector widget that can be either rectangle, ellipse, or polygon.

    Attributes
    ----------
    rois : List of dict
        A list of rois where each element is a dict with keys `region_specs` and 'region_type'. `region_specs` contain a tuple
        with specifications for the chosen region type (see ``Roi``).
        The region_type is a string identifyer that can be either rectangle, ellipse, or polygon.
    """

    def __init__(self, ax, type='rectangle'):
        self.rois = []
        self.ax = ax

        if type == 'rectangle':
            self.selector = RectangleSelector(self.ax, self.selector_callback, drawtype='box', interactive=True)
            self.type = type

        elif type == 'ellipse':
            self.selector = EllipseSelector(self.ax, self.selector_callback)
            self.type = type

        elif type == 'polygon':
            self.selector = PolygonSelector(self.ax, self.p_selector_callback)
            self.type = type

        else:
            raise TypeError('Type {} is not defined.'.format(type))

        plt.connect('key_press_event', self.key_pressed_callback)

    def selector_callback(self, eclick, erelease):
        """ eclick and erelease are matplotlib events at press and release."""
        print('startposition: {}'.format((eclick.xdata, eclick.ydata)))
        print('endposition  : {}'.format((erelease.xdata, erelease.ydata)))

    def p_selector_callback(self, vertices):
        print('Vertices: {}'.format(vertices))

    def key_pressed_callback(self, event):
        print('Key pressed.')
        if event.key in ['R', 'r']:
            print('RectangleSelector activated.')
            self.selector = RectangleSelector(self.ax, self.selector_callback, drawtype='box', interactive=True)
            self.type = 'rectangle'

        elif event.key in ['E', 'e']:
            print('EllipseSelector activated.')
            self.selector = EllipseSelector(self.ax, self.selector_callback, drawtype='box', interactive=True)
            self.type = 'ellipse'

        elif event.key in ['T', 't']:
            print('PolygonSelector activated.')
            self.selector = PolygonSelector(self.ax, self.p_selector_callback)
            self.type = 'polygon'

        else:
            pass

        if event.key in ['+'] and self.selector.active and self.type == 'rectangle':
            print('Roi was added.')
            region_specs = (np.flip(self.selector.geometry[:, 0]), self.selector.extents[1]-self.selector.extents[0],
                            self.selector.extents[3]-self.selector.extents[2], 0.)
            self.rois.append({'region_specs': region_specs, 'region_type': self.type})
            print('rois: {}'.format(self.rois))

        elif event.key in ['+'] and self.selector.active and self.type == 'ellipse':
            print('Roi was added.')
            region_specs = (self.selector.center, self.selector.extents[1]-self.selector.extents[0],
                            self.selector.extents[3]-self.selector.extents[2], 0.)

            self.rois.append({'region_specs': region_specs, 'region_type': self.type})
            print('rois: {}'.format(self.rois))

        elif event.key in ['+'] and self.selector.active and self.type == 'polygon':
            print('Roi was added.')
            vertices_ = self.selector.verts.append(self.selector.verts[0])
            self.rois.append({'region_specs': vertices_, 'region_type': self.type})
            print('rois: {}'.format(self.rois))

        else:
            pass


class Roi:
    """
    Class for a region of interest on LocData (roi).

    Roi objects define a region of interest for a referenced LocData object.

    Parameters
    ----------
    reference : LocData object, dict or surepy.data.metadata_pb2 object, or None
        Reference to localization data for which the region of interests are defined. It can be a LocData object,
        a reference to a saved SMLM file, or None for indicating no specific reference.
        When referencing a saved SMLM file, reference must be a dict or surepy.data.metadata_pb2 with keys `file_path`
        and `file_type` for a path pointing to a localization file and an integer indicating the file type.
        The integer should be according to surepy.data.metadata_pb2.file_type.
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
    properties_for_roi : tuple of string
        Localization properties in LocData object on which the region selection will be applied (for instance the
        coordinate_labels).


    Attributes
    ----------
    reference : LocData object, surepy.data.metadata_pb2 object, or None
        Reference to localization data for which the region of interests are defined. It can be a LocData object,
        a reference to a saved SMLM file, or None for indicating no specific reference.
        When referencing a saved SMLM file, reference have attributes `file_path`
        and `file_type` for a path pointing to a localization file and an integer indicating the file type.
        The integer should be according to surepy.data.metadata_pb2.file_type.
    _region : RoiRegion or list of RoiRegion
        Object specifying the geometrical region of interest. In case a list of RoiRegion is provided it is the union
        that makes up the region of interest.
    properties_for_roi : tuple of string
        Localization properties in LocData object on which the region selection will be applied (for instance the
        coordinate_labels).
    """

    def __init__(self, reference=None, region_type='', region_specs=None, properties_for_roi=()):
        if isinstance(reference, dict):
            self.reference = metadata_pb2.Metadata()
            for key, value in reference.items():
                setattr(self.reference, key, value)
        elif isinstance(reference, metadata_pb2.Metadata):
            self.reference = metadata_pb2.Metadata()
            self.reference.MergeFrom(reference)
        else:
            self.reference = reference

        self._region = RoiRegion(region_type=region_type, region_specs=region_specs)
        self.properties_for_roi = properties_for_roi

    def __repr__(self):
        return f'Roi(reference={self.reference}, ' \
            f'region_type={self._region.region_type}, ' \
            f'region_specs={self._region.region_specs},' \
            f' properties_for_roi={self.properties_for_roi})'

    def to_yaml(self, path=None):
        """
        Save Roi object in yaml format.

        Parameters
        ----------
        path : str, Path object, or None
            Path for yaml file. If None a roi file path is generated from the metadata.
        """

        # prepare path
        if path is None and isinstance(self.reference, LocData):
            _file_path = Path(self.reference.meta.file_path)
            _roi_file = _file_path.stem + '_roi.yaml'
            _path = _file_path.with_name(_roi_file)
        elif path is None and isinstance(self.reference, metadata_pb2.Metadata):
            _file_path = Path(self.reference.file_path)
            _roi_file = _file_path.stem + '_roi.yaml'
            _path = _file_path.with_name(_roi_file)
        else:
            _path = Path(path)

        # prepare reference for yaml representation - reference to LocData cannot be represented
        if self.reference is None:
            reference_for_yaml = None
        elif isinstance(self.reference, LocData):
            if self.reference.meta.file_path:
                meta_ = metadata_pb2.Metadata()
                meta_.file_path = self.reference.meta.file_path
                meta_.file_type = self.reference.meta.file_type
                reference_for_yaml = json_format.MessageToJson(meta_, including_default_value_fields=False)
            else:
                warnings.warn('The localization data has to be saved and the file path provided, '
                              'or the reference is lost.', UserWarning)
                reference_for_yaml = None
        else:
            reference_for_yaml = json_format.MessageToJson(self.reference, including_default_value_fields=False)

        # prepare points for yaml representation - numpy.float has to be converted to float
        def nested_change(iterable, func):
            if isinstance(iterable, (list, tuple)):
                return [nested_change(x, func) for x in iterable]
            return func(iterable)

        region_specs_for_yaml = nested_change(self._region.region_specs, float)

        region_type_for_yaml = self._region.region_type
        properties_for_roi_for_yaml = self.properties_for_roi

        yaml = YAML()
        output = dict(reference=reference_for_yaml,
                      region_type=region_type_for_yaml,
                      region_specs=region_specs_for_yaml,
                      properties_for_roi=properties_for_roi_for_yaml)
        yaml.dump(output, _path)

    @classmethod
    def from_yaml(cls, path):
        """
        Read Roi object from yaml format.

        Parameters
        ----------
        path : str or Path object
            Path for yaml file.
        """
        yaml = YAML(typ='safe')
        with open(path) as file:
            yaml_output = yaml.load(file)

        if yaml_output['reference'] is not None:
            reference_ = metadata_pb2.Metadata()
            reference_ = json_format.Parse(yaml_output['reference'], reference_)
        else:
            reference_ = yaml_output['reference']

        region_type_ = yaml_output['region_type']
        region_specs_ = yaml_output['region_specs']
        properties_for_roi_ = yaml_output['properties_for_roi']

        return cls(reference=reference_, region_type=region_type_, region_specs=region_specs_, properties_for_roi=properties_for_roi_)

    def locdata(self, reduce=True):
        """
        Localization data according to roi specifications.

        The ROI is applied on locdata properties as specified in self.properties_for_roi or by taking the first
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
        if isinstance(self.reference, LocData):
            locdata = self.reference
        elif isinstance(self.reference, metadata_pb2.Metadata):
            locdata = io.load_locdata(self.reference.file_path, self.reference.file_type)
        else:
            raise AttributeError('Valid reference to locdata is missing.')

        if self.properties_for_roi:
            pfr = self.properties_for_roi
        else:
            pfr = locdata.coordinate_labels[0:self._region.dimension]

        points = locdata.data[pfr]
        indices_inside = self._region.contains(points)
        new_locdata = LocData.from_selection(locdata=locdata, indices=indices_inside)

        # finish
        if reduce:
            new_locdata.reduce()

        # meta
        return new_locdata


def select_by_drawing(locdata, type='rectangle', **kwargs):
    """
    Select region of interest from rendered image by drawing rois.

    Parameters
    ----------
    locdata : LocData object
        The localization data from which to select localization data.
    type : str
        rectangle (default), ellipse, or polygon specifying the selection widget to use.

    Other Parameters
    ----------------
    kwargs :
        kwargs as specified for render2D

    Returns
    -------
    list
        A list of Roi objects
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    render2D(locdata, ax=ax, show=False, **kwargs)
    selector = _MplSelector(ax, type=type)
    plt.show()
    roi_list = [Roi(reference=locdata, region_specs=roi['region_specs'],
                    region_type=roi['region_type']) for roi in selector.rois]
    return roi_list
