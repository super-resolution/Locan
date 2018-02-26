"""
This module provides methods for analysis.
"""
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from scipy import stats

from surepy.analysis.analysis_tools import Analysis



class Localization_property(Analysis):
    """
    Analyze localization property with respect to probability density or variation over specified index.

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
    parameter : dict
        A dictionary with all settings for the current computation.
    results : pandas DataFrame
        Selected property with index.
    meta : dict
        metadata

    """
    count = 0

    def __init__(self, locdata, meta=None, property='Intensity', index=None):
        super().__init__(locdata, meta=meta, property=property, index=index)

        self.property = property
        self.index = index


    def _compute_results(self, locdata, meta=None, property='Intensity', index=None):
        if index is None:
            return locdata.data[[property]]
        else:
            return locdata.data[[property, index]].set_index(index)


    def hist(self, ax, bins='auto', log=True, fit=True):
        """ Provide matplotlib axes showing results. """
        ax.hist(self.results.values, bins=bins, normed=True, log=log)
        ax.set(title = self.property,
               xlabel = self.property,
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

            attribute_loc = self.property + '_loc'
            attribute_scale = self.property + '_scale'
            self.attribute_loc = loc    # todo fix attribute name
            self.attribute_scale = scale # todo fix attribute name


    def plot(self, ax, window=1):
        """ Provide matplotlib axes showing results. """
        self.results.rolling(window=window, center=True).mean().plot(ax=ax, legend=False)
        # todo: check rolling on arbitrary index
        ax.set(title = '{0}({1})'.format(self.property, self.index),
               xlabel = self.index,
               ylabel = self.property
               )
        ax.text(0.1,0.9,
                "window = " + str(window),
                transform = ax.transAxes
                )