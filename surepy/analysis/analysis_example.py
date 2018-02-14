"""
This module provides methods for analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from surepy.analysis.analysis import Analysis



class Analysis_example(Analysis):
    """
    Compute some data and provide a plot and histogram with secondary data (e.g. from fitting plot or histogram).

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    param : int or float
        Some parameter.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    results : numpy array
        Distances between the nearest localizations in two consecutive frames.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    """
    count = 0

    def __init__(self, locdata, meta=None, param=None):
        super().__init__(locdata, meta=meta, param=param)

        self.secondary_param_1 = None
        self.secondary_param_2 = None


    def _compute_results(self, locdata, param=None):
        # some results
        dict = {'a': np.random.normal(size=10),
                'b': np.random.normal(size=10)}
        results = pd.DataFrame.from_dict(dict)
        return (results)


    def hist(self, ax, property='a', bins='auto', fit=True):
        """ Provide histogram as matplotlib axes object showing hist(results). """
        ax.hist(self.results[property].values, bins=bins, normed=True, log=False)
        ax.set(title = 'Normal Data',
               xlabel = property,
               ylabel = 'PDF'
               )

        # fit distributions:
        if fit:
            # MLE fit of normal distribution on data
            loc, scale = stats.norm.fit(self.results[property].values)

            # plot
            x_values = np.linspace(stats.norm.ppf(0.01, loc=loc, scale=scale), stats.norm.ppf(0.99, loc=loc, scale=scale), 100)
            ax.plot(x_values, stats.norm.pdf(x_values, loc=loc, scale=scale), 'r-', lw = 3, alpha = 0.6, label = 'norm pdf')
            ax.text(0.1, 0.9,
                    'center: ' + str(loc) + '\n' + 'sigma: ' + str(scale),
                    transform=ax.transAxes
                    )

            attribute_center = property + '_center'
            attribute_sigma = property + '_center'
            self.attribute_center = loc
            self.attribute_sigma = scale


    def plot(self, ax=None, property=None, window=1):
        """ Provide plot as matplotlib axes object showing the running average of results over window size. """
        if ax is None:
            pass #todo
        self.results.rolling(window=window, center=True).mean().plot(ax=ax, y=property, legend=False)
        ax.set(title = 'Normal Data',
               xlabel = 'index',
               ylabel = property
               )
        ax.text(0.1,0.9,
                "window = " + str(window),
                transform = ax.transAxes
                )


    def save_as_yaml(self):
        """ Save results in a YAML format, that can e.g. serve as Origin import."""
        raise NotImplementedError