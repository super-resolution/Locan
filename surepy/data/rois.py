"""

Region of interest.

This module provides functions for managing regions of interest in localization data.

A Roi as defined by the class Roi is an object that defines a region of interest within a specific localization
dataset. It is therefore related to region specifications and a unique LocData object.

The Roi object provides methods for saving all specifications to a roi file, for loading them, and for returning LocData
with localizations within the roi region.

The roi region is provided as a RoiRegion object or another class instance that provides methods for getting a
printable representation (that can also be saved in a yaml file), for returning a matplotlib patch that can be
shown in a graph, for finding points within the region.

As a result we propose the following classes:

class Roi():
    Parameters
    ----------
    same as attributes

    Attributes
    ----------
    reference : LocData object, str, or None
        Reference to localization data for which the region of interests are defined. It can be a LocData object,
        True for indicating reference to a saved SMLM file, or None for indicating no specific reference.
    region : RoiRegion or list of RoiRegion
        Object specifying the geometrical region of interest. In case a list of RoiRegion is provided it is the union
        that makes up the region of interest.
    properties : tuple of string
        Localization properties in LocData object on which the region will be projected (for instance the
        coordinate_labels).

    ***** the following could be incorporated in reference?! *****
    meta : dict or better surepy.data.metadata_pb2 object
        Dict with keys file_path and file_type for a path pointing to a localization file and an integer indicating the
        file type. The integer should be according to surepy.data.metadata_pb2.file_type.

    Methods
    -------
    to_yaml()
        save specifications to yaml file
    from_yaml() : class method
        load specifications from yaml file and create Roi object
    locdata()
        return LocData object with all localizations from within the region of interest.


"""

from pathlib import Path
import warnings
import time

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
from surepy.data.filter import select_by_region
from surepy.render import render2D


class RoiRegion():
    """
    Region object for application that use regions of interest.

    A region that defines a region of interest with methods for getting a printable representation (that can also be
    saved in a yaml file), for returning a matplotlib patch that can be shown in a graph, for finding points within
    the region.

    Parameters
    ----------
    region_specs : tuple
        1D rois are defined by the following tuple:
        * interval: (start, stop)
        2D rois are defined by the following tuples:
        * rectangle: ((corner_x, corner_y), width, height, angle)
        * ellipse: ((center_x, center_y), width, height, angle)
        * polygon: ((point1_x, point1_y), (point2_x, point2_y), ...)
        3D rois are defined by the following tuples:
        * cuboid: ((corner_x, corner_y, corner_z), length, width, height, angle_1, angle_2, angle_3)
        * ellipsoid: ((center_x, center_y, center_z), length, width, height, angle_1, angle_2, angle_3)
        * polyhedron: (...)
    type : str
        Type is a string indicating the roi shape.
        In 1D it can be `interval`.
        In 2D it can be either `rectangle`, `ellipse`, or closed `polygon`.
        In 3D it can be either `cuboid` or `ellipsoid` or `polyhedron` (not implemented yet).

    Attributes
    ----------
    region_specs : tuple
        Specifications for region
    type : str
        Type of region
    region : RoiRegion object
        RoiRegion instance for the specified region type.
    polygon : ndarray of tuples
        Array of points for a closed polygon approximating the region of interest in clockwise orientation. The first
        and last point must be identical.
    dimension : int
        Spatial dimension of region
    centroid : tuple of float
        Centroid coordinates
    max_distance : array-like of float
        maximum distance between any two points in the region
    region_measure : float
        hull measure, i.e. area or volume
    subregion_measure : float
        measure of the sub-dimensional region, i.e. circumference or surface

    Methods
    -------

    mpl_patch(axes)
        Add patch for the specified region to axes
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
        return self._region.as_artist()

class _RoiInterval:

    def __init__(self, region_specs):
        if len(region_specs) != 2:
            raise AttributeError('The shape of region_specs must be (2,).')
        else:
            self.region_specs = region_specs

    def __repr__(self):
        return (dict(region_type='interval', region_specs=self.region_specs))

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


class _RoiRectangle():

    def __init__(self, region_specs):
        if len(region_specs) != 4:
            raise AttributeError('The shape of region_specs must be (2,).')
        else:
            self.region_specs = region_specs

    def __repr__(self):
        return (dict(region_type='rectangle', region_specs=self.region_specs))

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

        return (center_x, center_y)

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


class _RoiEllipse():

    def __init__(self, region_specs):
        if len(region_specs) != 4:
            raise AttributeError('The shape of region_specs must be (2,).')
        else:
            self.region_specs = region_specs

    def __repr__(self):
        return (dict(region_type='ellipse', region_specs=self.region_specs))

    def contains(self, points):
        center, width, height, angle = self.region_specs

        cos_angle = np.cos(np.radians(-angle))
        sin_angle = np.sin(np.radians(-angle))

        xc = points[:,0] - center[0]
        yc = points[:,1] - center[1]

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle

        rad_cc = (xct**2/(width/2.)**2) + (yct**2/(height/2.)**2)

        inside_indices = np.where(rad_cc <1.)[0]
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


class _RoiPolygon():

    def __init__(self, region_specs):
        if region_specs[0] != region_specs[-1]:
            raise ValueError('First and last element of polygon must be identical.')
        else:
            self.region_specs = region_specs

    def __repr__(self):
        return (dict(region_type='polygon', region_specs=self.region_specs))

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
        D = pdist(self.region_specs[0:-1])
        return np.nanmax(D)

    @property
    def region_measure(self):
        polygon = shPolygon(self.region_specs)
        return polygon.area

    @property
    def subregion_measure(self):
        polygon = shPolygon(self.region_specs)
        return polygon.length


class _MplSelector():
    '''
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
        A list of rois where each element is a dict with keys `region_specs` and 'type'. `region_specs` contain a tuple
        with specifications for the chosen region type (see ``Roi``).
        Type is a string identifyer that can be either rectangle, ellipse, or polygon.

    '''

    def __init__(self, ax, type='rectangle'):
        self.rois = []

        if type=='rectangle':
            self.selector = RectangleSelector(ax, self.selector_callback, drawtype='box', interactive=True)
            self.type = type

        elif type=='ellipse':
            self.selector = EllipseSelector(ax, self.selector_callback)
            self.type = type

        elif type=='polygon':
            self.selector = PolygonSelector(ax, self.p_selector_callback)
            self.type = type

        else:
            raise TypeError('Type {} is not defined.'.format(type))

        plt.connect('key_press_event', self.key_pressed_callback)

    def selector_callback(self, eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        print('startposition: {}'.format((eclick.xdata, eclick.ydata)))
        print('endposition  : {}'.format((erelease.xdata, erelease.ydata)))

    def p_selector_callback(self, vertices):
        print('Vertices: {}'.format(vertices))

    def key_pressed_callback(self, event):
        print('Key pressed.')
        if event.key in ['R', 'r']:
            print('RectangleSelector activated.')
            self.selector = RectangleSelector(ax, self.selector_callback, drawtype='box', interactive=True)
            self.type = 'rectangle'

        elif event.key in ['E', 'e']:
            print('EllipseSelector activated.')
            self.selector = EllipseSelector(ax, self.selector_callback, drawtype='box', interactive=True)
            self.type = 'ellipse'

        elif event.key in ['T', 't']:
            print('PolygonSelector activated.')
            self.selector = PolygonSelector(ax, self.p_selector_callback)
            self.type = 'polygon'

        else:
            pass

        if event.key in ['+'] and self.selector.active and self.type=='rectangle':
            print('Roi was added.')
            region_specs = (self.selector.corners[0], self.selector.extents[1]-self.selector.extents[0],
                            self.selector.extents[3]-self.selector.extents[2], 0.)
            self.rois.append({'region_specs': region_specs, 'type':self.type})
            print('rois: {}'.format(self.rois))

        elif event.key in ['+'] and self.selector.active and self.type=='ellipse':
            print('Roi was added.')
            region_specs = (self.selector.center, self.selector.extents[1]-self.selector.extents[0],
                            self.selector.extents[3]-self.selector.extents[2], 0.)

            self.rois.append({'region_specs': region_specs, 'type': self.type})
            print('rois: {}'.format(self.rois))

        elif event.key in ['+'] and self.selector.active and self.type=='polygon':
            print('Roi was added.')
            self.rois.append({'region_specs': self.selector.verts, 'type': self.type})
            print('rois: {}'.format(self.rois))

        else:
            pass


class Roi():
    """
    Class for a region of interest on LocData (roi).

    Roi objects define a region of interest for a referenced LocData object.

    Parameters
    ----------
    reference : LocData object, str, or None
        Reference to localization data for which the region of interests are defined. It can be a LocData object,
        True for indicating reference to a saved SMLM file, or None for indicating no specific reference.
    region_specs : tuple
        2D rois are defined by the following tuples:
        * rectangle: ((corner_x, corner_y), width, height, angle)
        * ellipse: ((center_x, center_y), width, height, angle)
        * polygon: ((point1_x, point1_y), (point2_x, point2_y), ...)
    type : str
        Type is a string indicating the roi shape. In 2D it can be either rectangle, ellipse, or closed polygon.
        In 3D it can be either cuboid or ellipsoid.
    meta : dict
        Dict with keys file_path and file_type for a path pointing to a localization file and an integer indicating the
        file type. The integer should be according to surepy.data.metadata_pb2.file_type.


    Attributes
    ----------
    reference : LocData object, str, or None
        Reference to localization data for which the region of interests are defined. It can be a LocData object,
        True for indicating reference to a saved SMLM file, or None for indicating no specific reference.
    region_specs : tuple
        2D rois are defined by the following tuples:
        * rectangle: ((corner_x, corner_y), width, height, angle)
        * ellipse: ((center_x, center_y), width, height, angle)
        * polygon: ((point1_x, point1_y), (point2_x, point2_y), ...)
    type : str
        Type is a string indicating the roi shape.
        In 1D it can be `interval`.
        In 2D it can be either `rectangle`, `ellipse`, or closed `polygon`.
        In 3D it can be either `cuboid` or `ellipsoid` (not implemented yet).
    meta : dict
        Dict with keys file_path and file_type for a path pointing to a localization file and an integer indicating the
        file type. The integer should be according to surepy.data.metadata_pb2.file_type.
    """

    def __init__(self, reference=None, region_specs=(), type='rectangle', meta=None):

        self.reference = reference
        self.region_specs = region_specs
        self.type = type
        self.meta = metadata_pb2.Metadata()

        # meta
        if isinstance(reference, LocData) and reference.meta.file_path:
            self.meta.file_path = reference.meta.file_path
            self.meta.file_type = reference.meta.file_type
        else:
            pass

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(self.meta, key, value)
        else:
            self.meta.MergeFrom(meta)


    def __repr__(self):
        return f'Roi(reference={self.reference}, region_specs={self.region_specs}, type={self.type}, meta={self.meta})'


    def to_yaml(self, path=None):
        '''
        Save Roi object in yaml format.

        Parameters
        ----------
        path : str, Path object, or None
            Path for yaml file. If None a roi file path is generated from the metadata.
        '''

        # prepare path
        if path is None:
            _file_path = Path(self.meta.file_path)
            _roi_file = _file_path.stem + '_roi.yaml'
            _path = _file_path.with_name(_roi_file)
        else:
            _path = Path(path)

        # prepare points for yaml representation - numpy.float has to be converted to float
        if self.type == 'rectangle' or self.type == 'ellipse':
            region_specs_for_yaml = tuple((float(v) for v in self.region_specs[0])), \
                                    float(self.region_specs[1]), \
                                    float(self.region_specs[2]), \
                                    float(self.region_specs[3])
        elif self.type == 'polygon':
            region_specs_for_yaml = tuple(((float(c) for c in point) for point in self.region_specs))

        # prepare reference for yaml representation - reference to LocData cannot be represented
        if isinstance(self.reference, LocData):
            if self.reference.meta.file_path or self.meta.file_path:
                reference_for_yaml = True
            else:
                warnings.warn('The localization data has to be saved and the file path provided, '
                              'or the reference is lost.', UserWarning)
                reference_for_yaml = None
        else:
            reference_for_yaml = self.reference

        # Prepare meta
        meta_json = json_format.MessageToJson(self.meta, including_default_value_fields=False)

        yaml = YAML()
        yaml.dump([reference_for_yaml, region_specs_for_yaml, self.type, meta_json], _path)


    @classmethod
    def from_yaml(cls, path, meta=None):
        '''
        Read Roi object from yaml format.

        Parameters
        ----------
        path : str or Path object
            Path for yaml file.
        '''
        yaml = YAML(typ='safe')
        with open(path) as file:
            reference, region_specs, type, meta_yaml = yaml.load(file)

        # meta
        meta_ = metadata_pb2.Metadata()
        meta_ = json_format.Parse(meta_yaml, meta_)
        meta_.modification_date = int(time.time())
        meta_.history.add(name = 'Roi.from_yaml')

        if meta is None:
            pass
        elif isinstance(meta, dict):
            for key, value in meta.items():
                setattr(meta_, key, value)
        else:
            meta_.MergeFrom(meta)

        return cls(reference=reference, region_specs=region_specs, type=type, meta=meta_)


    def locdata(self):
        '''
        Localization data according to roi specifications.

        Returns
        -------
        LocData
            A new instance of LocData with all localizations within region of interest.
        '''
        # todo implement ellipse and polygon for 2D and 3D
        # todo modify meta for locdata

        if isinstance(self.reference, LocData):
            return select_by_region(locdata=self.reference, roi=self)
        elif self.reference is True:
            locdata = io.load_locdata(self.meta.file_path, self.meta.file_type)
            return select_by_region(locdata=locdata, roi=self)


def select_by_drawing(locdata, type='rectangle', **kwargs):
    # todo: use metadata from locdata for reference
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
    roi_list = [Roi(reference=locdata, region_specs=roi['region_specs'], type=roi['type']) for roi in selector.rois]
    return roi_list


# todo: rename parameter meta
def load_from_roi_file(path, meta=None):
    """
    Load data from a Roi file.

    Parameters
    ----------
    path : string or Path object
        File path for a Roi file to load.
    meta : tuple or None
        Metadata that could carry alternative file specification (dict with file_path and file_type as defined for Roi
        objects) from which locdata is loaded while applying given roi specifications.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    roi = Roi.from_yaml(path, meta)
    locdata = roi.locdata()
    locdata.meta.history.add(name='load_from_roi_file')
    return locdata
