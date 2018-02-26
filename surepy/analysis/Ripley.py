"""
This module provides methods for computing Ripley's k function.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import RipleysKEstimator

from surepy.analysis.analysis_tools import _init_meta, _update_meta, save_results


#### The algorithms

def _ripleys_h_function(locdata, radii=np.linspace(0, 100, 10)):

    if locdata.coordinate_labels == {'Position_x', 'Position_y', 'Position_z'}:
        raise NotImplementedError('Ripley\'s k function is only implemented for 2D data.')

    x_min, y_min, x_max, y_max = [float(n) for n in locdata.bounding_box.hull.flatten()]
    area = float(locdata.properties['Region_measure_bb'])

    RKest = RipleysKEstimator(area, x_max, y_max, x_min, y_min)

    res_data = RKest.Hfunction(data=locdata.coordinates, radii=radii, mode='none')
    # res_csr = RKest.poisson(radii)

    # return pd.DataFrame({'radius': radii, 'Ripley_h_data':res_data, 'Ripley_h_csr':res_csr})
    return pd.DataFrame({'radius': radii, 'Ripley_h_data': res_data})


# The base analysis class

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


# The specific analysis classes

class Ripleys_h_function(_Ripley):
    """
    Compute Ripley's h function.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    radii : array of float
        The radii at which Ripley's k function is computed.


    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    results : pandas data frame
        The number of localizations per frame or
        the number of localizations per frame normalized to region_measure(hull).
    meta : dict
        meta data
    """
    count = 0

    def __init__(self, locdata, meta=None, radii=np.linspace(0, 100, 10)):
        super().__init__(locdata, meta=meta, radii=radii)

    def compute(self):
        data = self.locdata
        self.results = _ripleys_h_function(locdata=data, **self.parameter)
        return self



#### Interface functions


def plot(self, ax=None, show=True):
    '''
    Provide plot of results as matplotlib axes object.
    '''
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    self.results.plot(x='radius', ax=ax)
    ax.set(title = 'Ripley\'s h function',
           xlabel = 'Radius',
           ylabel = 'Ripley\'s h function'
           )
    ax.text(0.1,0.9,
            "Maximum: " + 'not yet',
            transform = ax.transAxes
            )

    # show figure
    if show:  # this part is needed if anyone wants to modify the figure
        plt.show()

    return None
