"""

Compute localizations per frame.

"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from surepy.analysis.analysis_base import _Analysis


#### The algorithms

def _localizations_per_frame(data, norm=None):
    # normalization
    if norm is None:
        normalization_factor = 1
        series_name = 'number_localizations'
    elif isinstance(norm, str):
        normalization_factor = data.properties[norm]
        series_name = f'number_localizations / ' + norm
    elif isinstance(norm, (int, float)):
        normalization_factor = norm
        series_name = f'number_localizations / {norm}'
    else:
        raise TypeError('normalization should be None, a number or a valid property name.')

    series = data.data.groupby('frame').size() / normalization_factor
    series.name = series_name
    return series


# The specific analysis classes

class LocalizationsPerFrame(_Analysis):
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
    distribution_statistics : Distribution_fits object, None
        Distribution parameters derived from MLE fitting of results.
    '''
    count = 0

    def __init__(self, locdata=None, meta=None, norm=None):
        super().__init__(locdata=locdata, meta=meta, norm=norm)
        self.distribution_statistics = None

    def compute(self):
        data = self.locdata
        self.results = _localizations_per_frame(data=data, **self.parameter)
        return self

    def fit_distributions(self, **kwargs):
        """
        Fit probability density functions to the distributions of `loc_property` values in the results
        using MLE (scipy.stats).

        Parameters
        ----------
        loc_property : LocData property
            The property for which to fit an appropriate distribution; if None all plots are shown.
        """
        self.distribution_statistics = _DistributionFits(self)
        self.distribution_statistics.fit(**kwargs)

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

        # prepare plot
        self.results.rolling(window=window, center=True).mean().plot(ax=ax, **kwargs)

        ax.set(title=f'Localizations per Frame\n (window={window})',
               xlabel = 'frame',
               ylabel = self.results.name
               )

        # show figure
        if show:  # this part is needed if anyone wants to modify the figure
            plt.show()

        return None

    def hist(self, ax=None, show=True, fit=True, bins='auto', **kwargs):
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

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to matplotlib.pyplot.hist().
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.hist(self.results.values, bins=bins, density=True, log=False, **kwargs)
        ax.set(title = 'Localizations per Frame',
               xlabel = self.results.name,
               ylabel = 'PDF'
               )

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

# todo: add fit function

class _DistributionFits:
    """
    Handle for distribution fits.

    It holds the statistical parameters derived by fitting the result distributions using MLE (scipy.stats).
    Statistical parameters are defined as described in
    :ref:(https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html)

    Parameters
    ----------
    analyis_class : LocalizationPrecision object
        The analysis class with result data to fit.

    Attributes
    ----------
    analyis_class : LocalizationPrecision object
        The analysis class with result data to fit.
    loc_property : LocData property
        The property for which to fit an appropriate distribution
    distribution : str or scipy.stats distribution object
        Distribution model to fit.
    parameters :
    """
    def __init__(self, analysis_class):
        self.analysis_class = analysis_class
        self.loc_property = self.analysis_class.results.name
        self.distribution = None
        self.parameters = []

    def fit(self, distribution=stats.norm, **kwargs):
        """
        Fit scipy.stats.distribution to analysis_class.results[loc_property].

        If with_constraints is true we put the following constraints on the fit procedure:
        If distribution is expon then floc=np.min(self.analysis_class.results[self.loc_property].values).

        Parameters
        ----------
        distribution : str or scipy.stats distribution object
            Distribution model to fit.

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters are passed to the `scipy.stat.distribution.fit()` function.
        """
        self.distribution = distribution

        loc, scale = self.distribution.fit(self.analysis_class.results.values, **kwargs)
        self.parameters.extend([self.loc_property + '_center', self.loc_property + '_sigma'])
        setattr(self, self.loc_property + '_center', loc)
        setattr(self, self.loc_property + '_sigma', scale)


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
        _center, _sigma = self.parameter_dict().values()

        x_values = np.linspace(self.distribution.ppf(0.001, loc=_center, scale=_sigma),
                               self.distribution.ppf(0.999, loc=_center, scale=_sigma), 100)
        ax.plot(x_values, self.distribution.pdf(x_values, loc=_center, scale=_sigma), 'r-', lw=3, alpha=0.6,
                label=str(self.distribution) + ' pdf', **kwargs)
        if show:
            plt.show()

    def parameter_dict(self):
        """ Dictionary of fitted parameters. """
        if self.parameters is None:
            return None
        else:
            return {k: self.__dict__[k] for k in self.parameters}