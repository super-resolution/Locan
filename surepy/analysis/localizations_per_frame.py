"""
This module provides methods for analysis.
"""
import matplotlib.pyplot as plt

from surepy.analysis.analysis_tools import _init_meta, _update_meta


#### The algorithms

def _localizations_per_frame(data, normalization=None):
    ''' Algorithm to compute localizations per frame.'''
    if normalization is None:
        normalization_factor = 1
    elif isinstance(normalization, str):
        normalization_factor = data.properties[normalization]
    elif isinstance(normalization, (int, float)):
        normalization_factor = normalization
    else:
        raise TypeError('normalization should be None, a number or a valid property name.')

    return data.data.groupby('Frame').size() / normalization_factor


#### The analysis classes


class _Localizations_per_frame():
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

    def hist(self, ax=None, show=True, bins='auto'):
        return hist(self, ax, show, bins)


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

class Localizations_per_frame(_Localizations_per_frame):
    '''
    Compute localizations per frame.

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
    '''
    count = 0

    def __init__(self, locdata=None, meta=None, normalization=None):
        super().__init__(locdata=locdata, meta=meta, normalization=normalization)

    def compute(self):
        data = self.locdata
        self.results = _localizations_per_frame(data=data, **self.parameter)
        return self



#### Interface functions

def plot(self, ax=None, show=True, window=1):
    '''
    Provide plot as matplotlib axes object showing the running average of results over window size.
    '''
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    self.results.rolling(window=window, center=True).mean().plot(ax=ax)

    ax.set(title = 'Localizations per Frame',
           xlabel = 'Frame',
           ylabel = 'number of localizations'
           )

    ax.text(0.1,0.9,
            "window = " + str(window),
            transform = ax.transAxes
            )

    # show figure
    if show:  # this part is needed if anyone wants to modify the figure
        plt.show()

    return None


def hist(self, ax=None, show=True, bins='auto'):
    '''
    Provide histogram as matplotlib axes object showing hist(results).
    '''
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.hist(self.results.values, bins=bins, normed=True, log=False)
    ax.set(title = 'Localizations per Frame',
           xlabel = 'number of localizations',
           ylabel = 'PDF'
           )

    # show figure
    if show:  # this part is needed if anyone wants to modify the figure
        plt.show()

    return None