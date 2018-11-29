"""
This module provides methods for computing Ripley's k function.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from surepy.analysis.analysis_base import _init_meta, _update_meta, save_results


#### The algorithms

def _ripleys_k_function(points, radii, region_measure=1, other_points=None):
    """
    Compute Ripley's K function for two- or three-dimensional data at the given radii.

    Parameters
    ----------
    points : array
        2D or 3D points on which to estimate Ripley's K function.
    radii: array of float
        Radii at which to compute Ripley's k function.
    region_measure : float
        region measure (area or volume) for points.
    other_points : array or None
        2D or 3D points from which to estimate Ripley's K function (e.g. subset of points). For None other_points
        is set to points (default).

    Returns
    -------
    ripley : 1D array of floats
        Ripley's K function evaluated at input radii.
    """
    if other_points is None:
        other_points = points

    nn = NearestNeighbors(metric='euclidean').fit(points)

    ripley_0 = []
    for radius in radii:
        indices_list = nn.radius_neighbors(other_points, radius, return_distance=False)
        ripley_0.append(np.array([len(indices)-1 for indices in indices_list]).sum())

    ripley = region_measure / (len(points) * (len(other_points) - 1)) * np.array(ripley_0)

    return ripley


def _ripleys_l_function(points, radii, region_measure=1, other_points=None):
    """
    Evaluates Ripley's L function which is different for 2D and 3D data points. For parameter description
    see _ripleys_k_function.
    """
    if np.shape(points)[1] == 2:
        return np.sqrt(_ripleys_k_function(points, radii, region_measure, other_points) / np.pi)
    elif np.shape(points)[1] == 3:
        return np.cbrt(_ripleys_k_function(points, radii, region_measure, other_points) * 3 / 4 / np.pi)


def _ripleys_h_function(points, radii, region_measure=1, other_points=None):
    """
    Evaluates Ripley's H function. For parameter description
    see _ripleys_k_function.
    """

    return _ripleys_l_function(points, radii, region_measure, other_points) - radii


##### The base analysis class

class _Ripley():
    """
    The base class for specialized analysis classes to be used on LocData objects.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    kwargs :
        Parameter that are passed to the algorithm.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    locdata : LocData object
        Localization data.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : numpy array or pandas DataFrame
        Computed results.
    """
    count = 0

    def __init__(self, locdata, meta, **kwargs):
        self.__class__.count += 1

        self.locdata = locdata
        self.parameter = kwargs
        self.meta = _init_meta(self)
        self.meta = _update_meta(self, meta)
        self.results = None


    def __del__(self):
        """ updating the counter upon deletion of class instance. """
        self.__class__.count -= 1

    def __str__(self):
        """ Return results in a printable format."""
        return str(self.results)

    def save_results(self, path):
        return save_results(self, path)

    def plot(self, ax=None, show=True):
        return plot(self, ax, show)



    def compute(self):
        """ Apply analysis routine with the specified parameters on locdata and return results."""
        raise NotImplementedError

    def save(self, path):
        """ Save Analysis object."""
        raise NotImplementedError

    def load(self, path):
        """ Load Analysis object."""
        raise NotImplementedError

    def report(self, ax):
        """ Show a report about analysis results."""
        raise NotImplementedError


##### The specific analysis classes

class Ripleys_k_function(_Ripley):
    """
    Compute Ripley's K function for two- or three-dimensional data at the given radii.

    Parameters
    ----------
    locdata : LocData object
        Localization data with 2D or 3D coordinates on which to estimate Ripley's K function.
    radii: array of float
        Radii at which to compute Ripley's k function.
    region_measure : float or 'auto'
        Region measure (area or volume) for point region. For 'bb' the region measure of the bounding_box is used.
    other_locdata : LocData object
        Other localization data from which to estimate Ripley's K function (e.g. subset of points).
        For None other_points is set to points (default).

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : pandas data frame
        Data frame with radii as provided and Ripley's K function.
    """
    count = 0

    def __init__(self, locdata, meta=None, radii=np.linspace(0, 100, 10), region_measure='bb', other_locdata=None):
        super().__init__(locdata=locdata, radii=radii, region_measure=region_measure, other_locdata=other_locdata, meta=meta)

    def compute(self):
        points = self.locdata.coordinates

        # turn other_locdata into other_points
        new_parameter = {key: self.parameter[key] for key in self.parameter if key is not 'other_locdata'}
        if self.parameter['other_locdata'] is not None:
            other_points = self.parameter['other_locdata'].coordinates
        else:
            other_points = None

        # choose the right region_measure
        # todo: add other hull regions
        if self.parameter['region_measure'] is 'bb':
            region_measure = float(self.locdata.properties['Region_measure_bb'])
        else:
            region_measure = self.parameter['region_measure']

        ripley = _ripleys_k_function(points=points, radii=self.parameter['radii'],
                                           region_measure=region_measure, other_points=other_points)
        self.results = pd.DataFrame({'radius': self.parameter['radii'], 'Ripley_k_data': ripley})
        return self


class Ripleys_l_function(_Ripley):
    """
    Compute Ripley's L function for two- or three-dimensional data at the given radii.

    Parameters
    ----------
    locdata : LocData object
        Localization data with 2D or 3D coordinates on which to estimate Ripley's K function.
    radii: array of float
        Radii at which to compute Ripley's k function.
    region_measure : float or 'auto'
        Region measure (area or volume) for point region. For 'bb' the region measure of the bounding_box is used.
    other_locdata : LocData object
        Other localization data from which to estimate Ripley's K function (e.g. subset of points).
        For None other_points is set to points (default).

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : pandas data frame
        Data frame with radii as provided and Ripley's L function.
    """
    count = 0

    def __init__(self, locdata, meta=None, radii=np.linspace(0, 100, 10), region_measure='bb', other_locdata=None):
        super().__init__(locdata=locdata, radii=radii, region_measure=region_measure, other_locdata=other_locdata, meta=meta)

    def compute(self):
        points = self.locdata.coordinates

        # turn other_locdata into other_points
        new_parameter = {key: self.parameter[key] for key in self.parameter if key is not 'other_locdata'}
        if self.parameter['other_locdata'] is not None:
            other_points = self.parameter['other_locdata'].coordinates
        else:
            other_points = None

        # choose the right region_measure
        # todo: add other hull regions
        if self.parameter['region_measure'] is 'bb':
            region_measure = float(self.locdata.properties['Region_measure_bb'])
        else:
            region_measure = self.parameter['region_measure']

        ripley = _ripleys_l_function(points=points, radii=self.parameter['radii'],
                                           region_measure=region_measure, other_points=other_points)
        self.results = pd.DataFrame({'radius': self.parameter['radii'], 'Ripley_l_data': ripley})
        return self


class Ripleys_h_function(_Ripley):
    """
    Compute Ripley's H function for two- or three-dimensional data at the given radii.

    Parameters
    ----------
    locdata : LocData object
        Localization data with 2D or 3D coordinates on which to estimate Ripley's K function.
    radii: array of float
        Radii at which to compute Ripley's k function.
    region_measure : float or 'auto'
        Region measure (area or volume) for point region. For 'bb' the region measure of the bounding_box is used.
    other_locdata : LocData object
        Other localization data from which to estimate Ripley's K function (e.g. subset of points).
        For None other_points is set to points (default).

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : pandas data frame
        Data frame with radii as provided and Ripley's H function.
    Ripley_h_maximum : pandas data frame
        Data frame with radius and Ripley's H value for the radius at which the H function has its maximum.
    """
    count = 0

    def __init__(self, locdata, meta=None, radii=np.linspace(0, 100, 10), region_measure='bb', other_locdata=None):
        super().__init__(locdata=locdata, radii=radii, region_measure=region_measure, other_locdata=other_locdata, meta=meta)
        self._Ripley_h_maximum = None

    def compute(self):
        # reset secondary results
        self._Ripley_h_maximum = None

        points = self.locdata.coordinates

        # turn other_locdata into other_points
        new_parameter = {key: self.parameter[key] for key in self.parameter if key is not 'other_locdata'}
        if self.parameter['other_locdata'] is not None:
            other_points = self.parameter['other_locdata'].coordinates
        else:
            other_points = None

        # choose the right region_measure
        # todo: add other hull regions
        if self.parameter['region_measure'] is 'bb':
            region_measure = float(self.locdata.properties['Region_measure_bb'])
        else:
            region_measure = self.parameter['region_measure']

        ripley = _ripleys_h_function(points=points, radii=self.parameter['radii'],
                                           region_measure=region_measure, other_points=other_points)
        self.results = pd.DataFrame({'radius': self.parameter['radii'], 'Ripley_h_data': ripley})
        return self


    @property
    def Ripley_h_maximum(self):
        if self._Ripley_h_maximum is None:
            index = self.results['Ripley_h_data'].idxmax()
            self._Ripley_h_maximum = self.results.iloc[index]
            return self._Ripley_h_maximum
        else:
            return self._Ripley_h_maximum

    @Ripley_h_maximum.deleter
    def Ripley_h_maximum(self):
        self._Ripley_h_maximum = None


##### Interface functions


def plot(self, ax=None, show=True):
    '''
    Provide plot of results as matplotlib axes object.
    '''
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    self.results.plot(x='radius', ax=ax)

    if self.results.columns[0] == 'Ripley_k_data':
        title = 'Ripley\'s K function'
    elif self.results.columns[0] == 'Ripley_l_data':
        title = 'Ripley\'s L function'
    elif self.results.columns[0] == 'Ripley_h_data':
        title = 'Ripley\'s H function'
    else:
        title = None

    ax.set(title = title,
           xlabel = 'Radius',
           ylabel = self.results.columns[0]
           )

    # show figure
    if show:
        plt.show()

    return None
