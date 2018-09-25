'''

Methods for managing regions of interest.

'''

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, PolygonSelector, EllipseSelector
from ruamel.yaml import YAML

from surepy.render import render2D


class _MplSelector():
    '''
    Class to use matplotlib widgets (RectangleSelector, EllipseSelector or PolygonSelector) on render localizations data.

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


class Roi_manager():
    """
    Class to manage a collection of regions of interest from which new LocData objects can be generated.

    Attributes
    ----------
    rois : List of dict
        A list of rois where each roi is a dict with keys 'points' and 'type'. Points are a list of tuples
        representing 2D or 3D coordinates. Type is a string identifyer that can be either rectangle, ellipse, or polygon.
    """

    def __init__(self):
        self.rois = []

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

    def select_by_drawing(self, locdata, type='rectangle', **kwargs):
        """
        Select from rendered image by drawing rois.

        Parameters
        ----------
        locdata : LocData object
            The localization data from which to select localization data.
        type : str
            rectangle (default), ellipse, or polygon specifying the selection widget to use.
        kwargs :
            kwargs as specified for render2D
        """
        fig, ax = plt.subplots(nrows=1, ncols=1)
        render2D(locdata, ax=ax, show=False, **kwargs)
        selector = _MplSelector(ax, type=type)
        plt.show()
        self.rois = selector.rois

    def save(self, path):
        yaml = YAML()
        yaml.dump(self.rois, path)

    def load(self, path):
        yaml = YAML(typ='safe')
        with open(path) as file:
            self.rois = yaml.load(file)

