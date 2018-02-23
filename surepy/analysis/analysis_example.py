import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from surepy.analysis.analysis import Analysis


def algorithm_1(data=None, limits=(0, 10)):
    ''' Provides a list of data values. data is actually not used.'''
    results = [i for i in range(*limits)]  # some complicated algorithm
    return results


def algorithm_2(data=None, n_sample=100, seed=None):
    ''' Provides random normal distributed data. data is actually not used.'''
    np.random.seed(seed)
    dict = {'a': np.random.normal(size=n_sample),
            'b': np.random.normal(size=n_sample)}
    results = pd.DataFrame.from_dict(dict)
    return results


class Analysis_example(Analysis):
    '''
    Compute some data and provide a plot and histogram with secondary data (e.g. from fitting plot or histogram).

    This is a specialized analysis class implementing an example analysis routine. For illustrating the analysis procedure it only takes a LocData object, creates some random data as result and
    provides plots and a report of the results. For providing the random data two algorithms are available.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    algorithm : string
        name of algorithm
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
    algorithm : callable
        name of algorithm
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : numpy array or pandas DataFrame
        Computed results.
    '''
    count = 0

    def __init__(self, locdata, algorithm=algorithm_1, meta=None, **kwargs):
        super().__init__()
        self.locdata = locdata
        self.algorithm = algorithm
        self.parameter = kwargs
        self.meta = self._init_meta(meta=meta)

        self.results = None
        self.secondary_results = None

    def compute(self):
        data = self.locdata  # take certain elements from locdata
        self.results = self.algorithm(data=data, **self.parameter)  # some complicated algorithm
        return None

    def save_results(self, path):
        return self.results  # this should be saving the data

    def plot(self, ax=None, show=True):
        '''
        A specialized plot to give a standardized visualization of results.
        '''
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.plot(self.results)
        ax.set(title='Normal Data',
               xlabel=property,
               ylabel='PDF'
               )

        # show figure
        if show:  # this part is needed if anyone wants to modify the figure
            plt.show()

        return None

    def plot_2(self, ax=None, show=True, bins='auto', normed=True, log=False, fit=True):
        '''
        A specialized plot to give a standardized visualization of results - in this case a histogram of results.
        '''
        if ax is None:
            fig = plt.figure(figsize=(8, 3))
            ax = fig.subplots(nrows=1, ncols=2)
            plt.subplots_adjust(wspace=0)

        # create histogram on first axes
        hist, bins, _ = ax[0].hist(self.results.values, bins=bins, normed=normed, log=log,
                                   label=list(self.results))
        ax[0].set(title='Normal Data',
                  xlabel='property',
                  ylabel='PDF'
                  )

        # create legend and results text on second axes
        h, l = ax[0].get_legend_handles_labels()
        ax[1].legend(h, l,
                     loc='upper left',
                     bbox_to_anchor=(0, 1),
                     title='Legend',
                     frameon=False,
                     borderaxespad=0)

        ax[1].set_axis_off()

        # fit distributions
        if fit:
            self.plot_histogram_fit(ax=ax, show=False)

        # show figure
        if show:  # this part is needed if anyone wants to modify the figure
            plt.tight_layout()
            plt.show()

        return None

    def plot_histogram_fit(self, ax=None, show=True):
        '''
        A specialized plot to give a standardized visualization of results - in this case a histogram of results.
        '''
        if ax is None:
            fig = plt.figure(figsize=(8, 3))
            ax = fig.subplots(nrows=1, ncols=2)
            plt.subplots_adjust(wspace=0)

        # fit distributions
        loc, scale = self.fit_histogram(data=self.results['a'].values, id='a')

        # plot fit
        x_values = np.linspace(stats.norm.ppf(0.01, loc=loc, scale=scale),
                               stats.norm.ppf(0.99, loc=loc, scale=scale), 100)
        ax[0].plot(x_values, stats.norm.pdf(x_values, loc=loc, scale=scale), 'r-', lw=3, alpha=0.6,
                   label='norm pdf')

        # present fit results
        ax[1].text(0, 0.5, 'Fit Results:')
        ax[1].text(0, 0.5,
                   'center: ' + str(loc) + '\n' + 'sigma: ' + str(scale),
                   horizontalalignment='left',
                   verticalalignment='top',
                   transform=ax[1].transAxes,
                   clip_on=False
                   )

        ax[1].set_axis_off()

        # show figure
        if show:
            plt.tight_layout()
            plt.show()

        return None

    def fit_histogram(self, data, id):

        # MLE fit of distribution on data
        loc, scale = stats.norm.fit(data)

        attribute_center = id + '_center'
        attribute_sigma = id + '_sigma'
        self.attribute_center = loc
        self.attribute_sigma = scale

        return loc, scale

    # there will be other specific visualization methods for other analysis routines.

    def report(self, path=None, show=True):
        '''
        Provide a report that is either displayed or saved as pdf.
        The report is a figure summarizing all visual representations. It is arranged specifically for a particular
        analysis routine.
        '''
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.subplots(nrows=3, ncols=2)

        # provide the axes elements (i.e. the plots)
        self.plot(ax=ax[0][0], show=False)
        self.plot_2(ax=ax[1][0:2], show=False)

        # adjust figure layout
        plt.tight_layout()

        # save figure as pdf
        if path is not None:
            plt.savefig(fname=path, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1,
                        frameon=None)

        # show figure
        if show:
            plt.show()

        return None

