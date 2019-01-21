"""

Region of interest.

This module provides functions for managing regions of interest in localization data.

"""

from pathlib import Path
import warnings
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, PolygonSelector, EllipseSelector
from ruamel.yaml import YAML

from google.protobuf import json_format

from surepy.data import metadata_pb2
from surepy import LocData
import surepy.io.io_locdata as io
from surepy.data.filter import select_by_region
from surepy.render import render2D


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
        # todo: correct for different shapes
        # if self.type=='rectangle':
        #     region_specs_for_yaml = self.region_specs
        #     region_specs_for_yaml = tuple([float(pt) for pt in self.region_specs])
        # else:
        #     region_specs_for_yaml = self.region_specs
        region_specs_for_yaml = list((float(v) for v in self.region_specs[0])), \
                                float(self.region_specs[1]), float(self.region_specs[2]), float(self.region_specs[3])

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
