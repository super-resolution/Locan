"""

Analyze localization property.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import stats

from surepy.analysis.analysis_base import _Analysis


##### The algorithms

def _localization_property(locdata, property='Intensity', index=None):
    if index is None:
        results = locdata.data[[property]]
    else:
        results = locdata.data[[property, index]].set_index(index)

    return results


##### The specific analysis classes

class Localization_property(_Analysis):
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

        self.results.rolling(window=window, center=True).mean().plot(ax=ax, legend=False, **kwargs)
        # todo: check rolling on arbitrary index
        ax.set(title=f"{self.parameter['property']}({self.parameter['index']}) (window={window})",
               xlabel = self.parameter['index'],
               ylabel = self.parameter['property']
               )

        # show figure
        if show:
            plt.show()

        return None


    def hist(self, ax=None, show=True, bins='auto', log=True, fit=True, **kwargs):
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
        log : Bool
            Flag for plotting on a log scale.
        fit: Bool
            Flag indicating if distributions fit are shown.
        kwargs : dict
            parameters passed to matplotlib.pyplot.hist().
        """
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