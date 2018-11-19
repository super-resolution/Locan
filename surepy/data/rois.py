'''

Methods for managing regions of interest.

'''
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, PolygonSelector, EllipseSelector
from ruamel.yaml import YAML

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
        A list of rois where each element is a dict with keys 'points' and 'type'. Points are a list of tuples
        representing 2D or 3D coordinates. Type is a string identifyer that can be either rectangle, ellipse, or polygon.

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

        if event.key in ['E', 'e']:
            print('EllipseSelector activated.')
            self.selector = EllipseSelector(ax, self.selector_callback, drawtype='box', interactive=True)
            self.type = 'ellipse'

        if event.key in ['T', 't']:
            print('PolygonSelector activated.')
            self.selector = PolygonSelector(ax, self.p_selector_callback)
            self.type = 'polygon'

        if event.key in ['+'] and self.selector.active and (self.type=='rectangle' or self.type=='ellipse'):
            print('Roi was added')
            self.rois.append({'points':self.selector.extents, 'type':self.type})
            print('rois: {}'.format(self.rois))

        if event.key in ['+'] and self.selector.active and self.type=='polygon':
            print('Roi was added')
            self.rois.append({'points':self.selector.verts, 'type':self.type})
            print('rois: {}'.format(self.rois))


class Roi():
    """
    Class for a region of interest on LocData (roi).

    Roi objects define a region of interest for a referenced LocData object. Roi objects can be managed by the
    Roi_manager.

    Parameters
    ----------
    reference : LocData object or path object to localization file
        Reference to localization data for which the region of interests are defined.
    meta : Metadata protobuf message or dictionary
        Metadata about the current roi (currently only the file path).

    Attributes
    ----------
    reference : LocData object or path object
        Reference to localization data for which the region of interests are defined. It can be a LocData object or
        a path pointing to a localization file.
    points : list or tuple of tuples
        Points are a list of tuples representing 2D or 3D coordinates.
    type : str
        Type is a string indicating the roi shape. It can be either rectangle, ellipse, or polygon.
    """
    # todo: clearify valid localization files.
    # todo: use protobuf metadata

    def __init__(self, reference=None, points=(), type='rectangle'):
        if isinstance(reference, LocData):
            try:
                self.reference = reference.meta.file
            except AttributeError:
                self.reference = None
        elif reference is None:
            self.reference = reference
        elif isinstance(reference, str) or isinstance(reference, Path):
            self.reference = str(reference)
        else:
            raise AttributeError('Parameter for reference needs to be LocData object or path object or None.')

        self._locdata = reference if isinstance(reference, LocData) else None
        self.points = points
        self.type = type

    def save(self, path):
        _path = Path(path)
        yaml = YAML()
        yaml.dump([self.reference, self.points, self.type], _path)

    def load(self, path):
        yaml = YAML(typ='safe')
        with open(path) as file:
            self.reference, self.points, self.type = yaml.load(file)
        self._locdata = self.reference if isinstance(self.reference, LocData) else None

    def locdata(self):
        new_locdata = None
        return new_locdata


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
    roi_list = [Roi(reference=locdata, points=roi['points'], type=roi['type']) for roi in selector.rois]
    return roi_list





class Roi_manager():
    """
    Class to manage a collection of regions of interest from which new LocData objects can be generated.

    Parameters
    ----------
    rois : List of Roi objects
        A list of rois specified as Roi objects.

    Attributes
    ----------
    rois : List of Roi objects
        A list of rois specified as Roi objects.

    """


    def __init__(self, rois=None):
        self.rois = rois

    @property
    def locdata(self):
       """ Return the LocData object from which all rois are derived. """
       if isinstance(self.reference, LocData):
           locdata = self.reference

       elif isinstance(self.reference, str) or isinstance(self.reference, Path):
           path = Path(self.reference)
           locdata = io.load_rapidSTORM_file(path)

       else:
           raise AttributeError('No reference to LocData or file path is given.')

       return locdata


    @property
    def locdatas(self):
       """ Return a list with LocData objects for all specified rois. """
       if isinstance(self.reference, LocData):
           locdatas = [select_by_region(self.reference, roi) for roi in self.rois]

       elif isinstance(self.reference, str) or isinstance(self.reference, Path):
           path = Path(self.reference)
           locdata = io.load_rapidSTORM_file(path)
           locdatas = [select_by_region(locdata, roi) for roi in self.rois]

       else:
           raise AttributeError('No reference to LocData or file path is given.')

       return locdatas


    def add_rectangle(self, extents):
        roi_dict = {'points': extents, 'type': 'rectangle'}
        self.rois.append(roi_dict)

    def add_ellipse(self, extents):
        roi_dict = {'points': extents, 'type': 'ellipse'}
        self.rois.append(roi_dict)

    def add_polygone(self, vertices):
        roi_dict = {'points': vertices, 'type': 'polygon'}
        self.rois.append(roi_dict)

    def add_rectangles(self, extents):
        roi_list = []
        for element in extents:
            roi_dict = {'points': element, 'type': 'rectangle'}
            roi_list.append(roi_dict)
        self.rois += roi_list

    def add_ellipses(self, extents):
        roi_list = []
        for element in extents:
            roi_dict = {'points': element, 'type': 'ellipse'}
            roi_list.append(roi_dict)
        self.rois += roi_list

    def add_polygons(self, vertices_list):
        roi_list = []
        for vertices in vertices_list:
            roi_dict = {'points': vertices, 'type': 'polygon'}
            roi_list.append(roi_dict)
        self.rois += roi_list

    def clear(self):
        self.rois = []





