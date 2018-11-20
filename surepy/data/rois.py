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
    reference : LocData object, str, or None
        Reference to localization data for which the region of interests are defined. It can be a LocData object,
        "FILE" for indicating reference to a saved SMLM file, or None for indicating no specific reference.
    points : tuple or tuple of tuples
        Points are a tuple with extends (i.e. min and max values for each coordinate) for rectangle or ellipse.
        It is a tuple or list of tuples representing 2D or 3D coordinates (i.e. vertices) for polygon.
    type : str
        Type is a string indicating the roi shape. It can be either rectangle, ellipse, or polygon.
    meta : dict
        Dict with keys file_path and file_type for a path pointing to a localization file and a string indicating the
        file type. The string should be according to locdata metadata.


    Attributes
    ----------
    reference : LocData object, str, or None
        Reference to localization data for which the region of interests are defined. It can be a LocData object,
        "FILE" for indicating reference to a saved SMLM file, or None for indicating no specific reference.
    points : tuple or tuple of tuples
        Points are a tuple with extends (i.e. min and max values for each coordinate) for rectangle or ellipse.
        It is a tuple or list of tuples representing 2D or 3D coordinates (i.e. vertices) for polygon.
    type : str
        Type is a string indicating the roi shape. It can be either rectangle, ellipse, or polygon.
    meta : dict
        Dict with keys file_path and file_type for a path pointing to a localization file and a string indicating the
        file type. The string should be according to locdata metadata.
    """
    # todo: use protobuf metadata
    # todo: include meta : Metadata protobuf message or dictionary : Metadata about the current roi (currently only the file path).

    def __init__(self, reference=None, points=(), type='rectangle', meta=None):

        self.reference = reference
        self.points = points
        self.type = type
        self.meta = meta

        if isinstance(reference, LocData):
            meta_dict = dict(file_path=reference.meta.file_path, file_type=reference.meta.file_path)
            self.meta = meta_dict


    def __repr__(self):
        return f'Roi(reference={self.reference}, points={self.points}, type={self.type}, meta={self.meta})'


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
            _file_path = Path(self.meta['file_path'])
            _roi_file = _file_path.stem + '_roi.yaml'
            _path = _file_path.with_name(_roi_file)
            self.meta['roi_path'] = str(_path)
        else:
            _path = Path(path)
            self.meta['roi_path'] = str(_path)

        # prepare floats for yaml representation
        # todo: correct for different shapes
        if self.type=='rectangle':
            points_for_yaml = tuple([float(pt) for pt in self.points])
        else:
            points_for_yaml = self.points

        yaml = YAML()
        yaml.dump([self.reference, points_for_yaml, self.type, self.meta], _path)


    def from_yaml(self, path):
        '''
        Read Roi object from yaml format.

        Parameters
        ----------
        path : str or Path object
            Path for yaml file.
        '''
        yaml = YAML(typ='safe')
        with open(path) as file:
            self.reference, self.points, self.type, self.meta = yaml.load(file)


    def locdata(self, **kwargs):
        '''
        Localization data according to roi specifications.

        Parameters
        ----------
        kwargs :
            kwargs valid for Locdata.from_selection()

        Returns
        -------
        LocData
            A new instance of LocData with all localizations within region of interest.
        '''
        # todo implement ellipse and polygon for 2D and 3D

        if isinstance(self.reference, LocData):
            return select_by_region(self.reference, self, **kwargs)
        elif self.reference=='FILE':
            locdata =  io.load_locdata(self.meta['file_path'], self.meta['file_type'])
            return select_by_region(locdata, self, **kwargs)



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
