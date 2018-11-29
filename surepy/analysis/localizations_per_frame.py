"""

Compute localizations per frame.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surepy.analysis.analysis_base import _Analysis


#### The algorithms

def _localizations_per_frame(data, norm=None):
    # normalization
    if norm is None:
        normalization_factor = 1
    elif isinstance(norm, str):
        normalization_factor = data.properties[norm]
    elif isinstance(norm, (int, float)):
        normalization_factor = norm
    else:
        raise TypeError('normalization should be None, a number or a valid property name.')

    return data.data.groupby('Frame').size() / normalization_factor


# The specific analysis classes

class Localizations_per_frame(_Analysis):
    '''
    Compute localizations per frame.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    data : Pandas DataFrame
        DataFrame object that contains a column `Frame` to be grouped.
    norm : int, float, str, None
        Normalization factor that can be None, a number, or another property in `data`.

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

    def __init__(self, locdata=None, meta=None, norm=None):
        super().__init__(locdata=locdata, meta=meta, norm=norm)

    def compute(self):
        data = self.locdata
        self.results = _localizations_per_frame(data=data, **self.parameter)
        return self

    def plot(self, ax=None, show=True, window=1, **kwargs):
        """
        Provide plot as matplotlib axes object showing the running average of results over window size.

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image
        show : bool
            Flag indicating if plt.show() is active.
        window: int
            Window for running average that is applied before plotting.
        kwargs : dict
            parameters passed to matplotlib.pyplot.plot().
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # prepare plot
        self.results.rolling(window=window, center=True).mean().plot(ax=ax, **kwargs)

        ax.set(title=f'Localizations per Frame (window={window})',
               xlabel = 'Frame',
               ylabel = 'number of localizations'
               )

        # show figure
        if show:  # this part is needed if anyone wants to modify the figure
            plt.show()

        return None

    def hist(self, ax=None, show=True, bins='auto', **kwargs):
        """
        Provide histogram as matplotlib axes object showing hist(results).

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image
        show : bool
            Flag indicating if plt.show() is active.
        bins : float
            Bin specifications (passed to matplotlib.hist).
        fit: Bool
            Flag indicating if distributions fit are shown.
        kwargs : dict
            parameters passed to matplotlib.pyplot.hist().
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.hist(self.results.values, bins=bins, normed=True, log=False, **kwargs)
        ax.set(title = 'Localizations per Frame',
               xlabel = 'number of localizations',
               ylabel = 'PDF'
               )

        # show figure
        if show:
            plt.show()

        return None

# todo: add fit function
