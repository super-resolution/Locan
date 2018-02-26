"""
This module provides methods for analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import stats

from surepy.analysis.analysis_tools import _init_meta, _update_meta, save_results


#### The algorithms

def _localization_property(locdata, property='Intensity', index=None):
    if index is None:
        results = locdata.data[[property]]
    else:
        results = locdata.data[[property, index]].set_index(index)

    return results


# The base analysis class

class _Localization_property():
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

    def plot(self, ax=None, show=True, window=1):
        return plot(self, ax, show, window)

    def hist(self, ax=None, show=True, bins='auto', log=True, fit=True):
        return hist(self, ax, show, bins, log, fit)


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

class Localization_property(_Localization_property):
    '''
    Analyze localization property with respect to probability density or variation over a specified index.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    property : str
        The property to analyze.
    index : str or None
        The property name that should serve as index (i.e. x-axis in x-y-plot)

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
    '''
    count = 0

    def __init__(self, locdata=None, meta=None, property='Intensity', index=None):
        super().__init__(locdata=locdata, meta=meta, property=property, index=index)

    def compute(self):
        data = self.locdata
        self.results = _localization_property(locdata=data, **self.parameter)
        return self



#### Interface functions


def plot(self, ax=None, show=True, window=1):
    """ Provide matplotlib axes showing results. """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    self.results.rolling(window=window, center=True).mean().plot(ax=ax, legend=False)
    # todo: check rolling on arbitrary index
    ax.set(title = '{0}({1})'.format(self.parameter['property'], self.parameter['index']),
           xlabel = self.parameter['index'],
           ylabel = self.parameter['property']
           )
    ax.text(0.1,0.9,
            "window = " + str(window),
            transform = ax.transAxes
            )

    # show figure
    if show:
        plt.show()

    return None


def hist(self, ax=None, show=True, bins='auto', log=True, fit=True):
    """ Provide matplotlib axes showing results. """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.hist(self.results.values, bins=bins, normed=True, log=log)
    ax.set(title = self.parameter['property'],
           xlabel = self.parameter['property'],
           ylabel = 'PDF'
           )

    # fit distributions:
    if fit:
        # MLE fit of exponential distribution
        loc, scale = stats.expon.fit(self.results.values, floc=np.min(self.results.values))

        # plot
        x_values = np.linspace(stats.expon.ppf(0.001, loc=loc, scale=scale),
                               stats.expon.ppf(0.999, loc=loc, scale=scale), 100)
        ax.plot(x_values, stats.expon.pdf(x_values, loc=loc, scale=scale), 'r-', lw = 3, alpha = 0.6,
                label = 'exponential pdf')
        ax.text(0.1, 0.8,
                'loc: ' + str(loc) + '\n' + 'scale: {:3.2f}'.format(scale),
                transform=ax.transAxes
                )

        attribute_loc = self.parameter['property'] + '_loc'
        attribute_scale = self.parameter['property'] + '_scale'
        self.attribute_loc = loc    # todo fix attribute name
        self.attribute_scale = scale # todo fix attribute name

    # show figure
    if show:
        plt.show()

    return None