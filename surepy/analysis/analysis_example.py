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


    def hist(self, ax=None, show=True, property='a', bins='auto', fit=True):
        """ Provide histogram as matplotlib axes object showing hist(results). """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

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

            # set attributes with secondary results
            attribute_center = property + '_center'
            attribute_sigma = property + '_center'
            self.attribute_center = loc
            self.attribute_sigma = scale

        #show figure
        if show:  # this part is needed if anyone wants to modify the figure
            plt.show()


    def plot(self, ax=None, show=True, property=None, window=1):
        """ Provide plot as matplotlib axes object showing the running average of results over window size. """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        self.results.rolling(window=window, center=True).mean().plot(ax=ax, y=property, legend=False)
        ax.set(title = 'Normal Data',
               xlabel = 'index',
               ylabel = property,
               label = self.meta.identifier
               )
        ax.text(0.1,0.9,
                "window = " + str(window),
                transform = ax.transAxes
                )
        ax.legend()

        #show figure
        if show:  # this part is needed if anyone wants to modify the figure
            plt.show()


    def report(self, path=None, show=True):
        '''
        Provide a report that is either displayed or saved as pdf.

        Parameter
        ---------
        path : string or Path object
            File path for a report file. If path is None the report will be displayed.

        Returns
        -------

        '''
        # instantiate a figure with axes elements
        fig, ax = plt.subplots(nrows=1, ncols=3)

        # provide the axes elements (i.e. the plots)
        self.plot(ax=ax[0])
        self.hist(ax=ax[1])
        ax[2].text(0.8, 0.9,
                  "some text",
                  transform=ax[2].transAxes
                  )

        # adjust figure layout
        plt.tight_layout()

        # save figure as pdf
        if path is not None:
            plt.savefig(fname=path, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1,
                        frameon=None)

        #show figure
        if show:
            plt.show()

