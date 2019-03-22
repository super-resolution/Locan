"""
Example for a specialized analysis class.

It includes two algorithms for specific analysis routines.
And it provides standard interface functions modified for the specific analysis routine like report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from surepy.analysis.analysis_base import _Analysis

#
#### The algorithms
#
# First there is an algorithm to compute something given a set of point coordinates.
# Second there might be an alternative algorithm doing the same thing.
# Both can be tested with some simple point data.
#
# The user will not have to use this particular function directly.
#

def _algorithm_1(data=None, limits=(0, 10)):
    ''' Provides a list of data values. data is actually not used.'''
    results = [i for i in range(*limits)]  # some complicated algorithm
    return results


def _algorithm_2(data=None, n_sample=100, seed=None):
    ''' Provides random normal distributed data. data is actually not used.'''
    np.random.seed(seed)
    dict = {'a': np.random.normal(size=n_sample),
            'b': np.random.normal(size=n_sample)}
    results = pd.DataFrame.from_dict(dict)
    return results

#
##### The base analysis class
#
# Now we want a class implementing this algorithm to be used with locdata. Also the results from this algorithm should
# be reused in some visual representation that is specific for this analysis routine.
# Therefore we have a class that holds results and organizes metadata and provides the specific plotting routine.
#

# class: analysis.analysis_base._Analysis


#
# This specific analysis classes inherit from _Analysis.
#
# The classes for each particular algorith are defined as:
#


#
# This specific analysis classes inherit from _Analysis.
#
# The classes for each particular algorith are defined as:
#

class AnalysisExampleAlgorithm_1(_Analysis):
    '''
    Example for an analysis class implementing algorithm_2.
    Compute some data and provide a plot and histogram with secondary data (e.g. from fitting plot or histogram).

    This is a specialized analysis class implementing an example analysis routine. For illustrating the analysis
    procedure it only takes a LocData object, creates some random data as result and
    provides plots and a report of the results.

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

    def __init__(self, locdata=None, meta=None, limits=(0, 10)):
        super().__init__(locdata=locdata, meta=meta, limits=limits)

    def compute(self):
        data = self.locdata  # take certain elements from locdata
        self.results = _algorithm_1(data=data, **self.parameter)  # some complicated algorithm
        return self



class AnalysisExampleAlgorithm_2(_Analysis):
    '''
    Example for an analysis class implementing algorithm_2.
    Compute some data and provide a plot and histogram with secondary data (e.g. from fitting plot or histogram).

    This is a specialized analysis class implementing an example analysis routine. For illustrating the analysis
    procedure it only takes a LocData object, creates some random data as result and
    provides plots and a report of the results.

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

    def __init__(self, locdata=None, meta=None, n_sample=100, seed=None):
        super().__init__(locdata=locdata, meta=meta, n_sample=n_sample, seed=seed)

    def compute(self):
        data = self.locdata  # take certain elements from locdata
        self.results = _algorithm_2(data=data, **self.parameter)  # some complicated algorithm
        return self

#
#### Interface functions
#
# Now we have a class structure with results that can be further processed.
# Secondary results from e.g. fit procedures are added to the analysis class as new attributes.
#


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
        plot_histogram_fit(self, ax=ax, show=False)

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
    loc, scale = fit_histogram(self, data=self.results['a'].values, id='a')

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
    self.hist(ax=ax[1][0:2], show=False)

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

