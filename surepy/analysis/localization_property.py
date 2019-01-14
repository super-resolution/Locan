"""

Analyze localization property.

Localizations come with a range of properties including position coordinates, emission strength, local background etc..
Most properties represent random variables that were drawn from an unknown probability distribution.
It is often useful to analyze the properties from all localizations within a selection and estimate the corresponding
probability distribution.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import stats

from surepy.analysis.analysis_base import _Analysis, _list_parameters


##### The algorithms

def _localization_property(locdata, loc_property='Intensity', index=None):
    if index is None:
        results = locdata.data[[loc_property]]
    else:
        results = locdata.data[[loc_property, index]].set_index(index)

    return results


##### The specific analysis classes

class Localization_property(_Analysis):
    """
    Analyze localization property with respect to probability density or variation over a specified index.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    loc_property : str
        The property to analyze.
    index : str or None
        The property name that should serve as index (i.e. x-axis in x-y-plot)

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    locdata : LocData object
        Localization data.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : numpy array or pandas DataFrame
        Computed results.
    distribution_statistics : Distribution_stats object, None
        Distribution parameters derived from MLE fitting of results.
    """
    def __init__(self, locdata=None, meta=None, loc_property='Intensity', index=None):
        super().__init__(locdata=locdata, meta=meta, loc_property=loc_property, index=index)
        self.distribution_statistics = None

    def compute(self):
        data = self.locdata
        self.results = _localization_property(locdata=data, **self.parameter)
        return self

    def fit_distributions(self, distribution=stats.expon, with_constraints=True, **kwargs):
        """
        Fit probability density functions to the distributions of `loc_property` values in the results
        using MLE (scipy.stats).

        If with_constraints is true we put the following constraints on the fit procedure:
        If distribution is expon then `floc=np.min(self.analysis_class.results[self.loc_property].values)`.

        Parameters
        ----------
        distribution : str or scipy.stats distribution object
            Distribution model to fit.
        with_constraints : bool
            Flag to use predefined constraints on fit parameters.

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters are passed to the `scipy.stat.distribution.fit()` function.
        """
        self.distribution_statistics = _DistributionFits(self)
        self.distribution_statistics.fit(distribution, with_constraints=with_constraints, **kwargs)


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

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to matplotlib.pyplot.plot().
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        self.results.rolling(window=window, center=True).mean().plot(ax=ax, legend=False, **kwargs)
        # todo: check rolling on arbitrary index
        ax.set(title=f"{self.parameter['loc_property']}({self.parameter['index']})\n (window={window})",
               xlabel = self.parameter['index'],
               ylabel = self.parameter['loc_property']
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
        fit: Bool or None
            Flag indicating if distribution fit is shown. The fit will only be computed if `distribution_statistics`
             is None.

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to matplotlib.pyplot.hist().
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.hist(self.results.values, bins=bins, density=True, log=log, **kwargs)
        ax.set(title = self.parameter['loc_property'],
               xlabel = self.parameter['loc_property'],
               ylabel = 'PDF'
               )

        # fit distributions:
        if fit:
            if isinstance(self.distribution_statistics, _DistributionFits):
                self.distribution_statistics.plot(ax=ax, show=False)
            else:
                self.fit_distributions()
                self.distribution_statistics.plot(ax=ax, show=False)

        # show figure
        if show:
            plt.show()

        return None

# todo add Dependence_stats to fit a plot to a linear function, log function, or exponential decay.

class _DistributionFits:
    """
    Handle for distribution fits.

    This class is typically instantiated by Localization_property methods.
    It holds the statistical parameters derived by fitting the result distributions using MLE (scipy.stats).
    Statistical parameters are defined as described in
    :ref:(https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html)

    Parameters
    ----------
    analyis_class : Localization_precision object
        The analysis class with result data to fit.

    Attributes
    ----------
    analyis_class : Localization_precision object
        The analysis class with result data to fit.
    loc_property : LocData property
        The property for which to fit an appropriate distribution
    distribution : str or scipy.stats distribution object
        Distribution model to fit.
    parameters : list of string
        Distribution parameters.
    """
    def __init__(self, analysis_class):
        self.analysis_class = analysis_class
        self.loc_property = self.analysis_class.parameter['loc_property']
        self.distribution = None
        self.parameters = []

    def fit(self, distribution, with_constraints=True, **kwargs):
        """
        Fit scipy.stats.distribution to analysis_class.results[loc_property].

        If with_constraints is true we put the following constraints on the fit procedure:
        If distribution is expon then floc=np.min(self.analysis_class.results[self.loc_property].values).

        Parameters
        ----------
        distribution : str or scipy.stats distribution object
            Distribution model to fit.
        with_constraints : bool
            Flag to use predefined constraints on fit parameters.

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters are passed to the `scipy.stat.distribution.fit()` function.
        """
        self.distribution = distribution
        for param in _list_parameters(distribution):
            self.parameters.append(self.loc_property +  '_' + param)

        if with_constraints and self.distribution == stats.expon:
            # MLE fit of exponential distribution with constraints
            fit_results = stats.expon.fit(self.analysis_class.results[self.loc_property].values,
                                         floc=np.min(self.analysis_class.results[self.loc_property].values), **kwargs)
            for parameter, result in zip(self.parameters, fit_results):
                setattr(self, parameter, result)
        else:
            fit_results = self.distribution.fit(self.analysis_class.results[self.loc_property].values, **kwargs)
            for parameter, result in zip(self.parameters, fit_results):
                setattr(self, parameter, result)


    def plot(self, ax=None, show=True, **kwargs):
        """
        Provide plot as matplotlib axes object showing the probability distribution functions of fitted results.

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image.
        show : bool
            Flag indicating if plt.show() is active.

        Other Parameters
        ----------------
        kwargs : dict
            parameters passed to matplotlib.pyplot.plot().
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # plot fit curve
        parameter = self.parameter_dict().values()
        x_values = np.linspace(self.distribution.ppf(0.001, *parameter),
                               self.distribution.ppf(0.999, *parameter), 100)
        ax.plot(x_values, self.distribution.pdf(x_values, *parameter), 'r-', lw=3, alpha=0.6,
                label=str(self.distribution) + ' pdf', **kwargs)
        if show:
            plt.show()

    def parameter_dict(self):
        """ Dictionary of fitted parameters. """
        return {k: self.__dict__[k] for k in self.parameters}


