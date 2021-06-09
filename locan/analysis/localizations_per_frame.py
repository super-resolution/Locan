"""

Compute localizations per frame.

"""
import logging

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from locan.analysis.analysis_base import _Analysis


__all__ = ['LocalizationsPerFrame']

logger = logging.getLogger(__name__)


#### The algorithms

def _localizations_per_frame(data, norm=None):
    """
    Compute localizations per frame.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame object that contains a column `Frame` to be grouped.
    norm : int, float, str, None
        Normalization factor that can be None, a number, or another property in `data`.

    Returns:
    --------
    pandas.Series
    """
    # normalization
    if norm is None:
        normalization_factor = 1
        series_name = 'n_localizations'
    elif isinstance(norm, str):
        normalization_factor = data.properties[norm]
        series_name = f'n_localizations / ' + norm
    elif isinstance(norm, (int, float)):
        normalization_factor = norm
        series_name = f'n_localizations / {norm}'
    else:
        raise TypeError('normalization should be None, a number or a valid property name.')

    series = data.data.groupby('frame').size() / normalization_factor
    series.name = series_name
    return series


# The specific analysis classes

class LocalizationsPerFrame(_Analysis):
    """
    Compute localizations per frame.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    norm : int, float, str, None
        Normalization factor that can be None, a number, or another property in `data`.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : numpy.ndarray, pandas.DataFrame
        Computed results.
    distribution_statistics : Distribution_fits object, None
        Distribution parameters derived from MLE fitting of results.
    """
    count = 0

    def __init__(self, meta=None, norm=None):
        super().__init__(meta=meta, norm=norm)
        self.results = None
        self.distribution_statistics = None

    def compute(self, locdata):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
           Localization data.

        Returns
        -------
        Analysis class
           Returns the Analysis class object (self).
        """
        if not len(locdata):
            logger.warning('Locdata is empty.')
            return self

        self.results = _localizations_per_frame(data=locdata, **self.parameter)
        return self

    def fit_distributions(self, **kwargs):
        """
        Fit probability density functions to the distributions of `loc_property` values in the results
        using MLE (scipy.stats).

        Parameters
        ----------
        loc_property : str
            The LocData property for which to fit an appropriate distribution; if None all plots are shown.
        """
        if self:
            self.distribution_statistics = _DistributionFits(self)
            self.distribution_statistics.fit(**kwargs)
        else:
            logger.warning('No results available to fit.')

    def plot(self, ax=None, window=1, **kwargs):
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the running average of results over window size.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which to show the image
        window: int
            Window for running average that is applied before plotting.
        kwargs : dict
            Other parameters passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        # prepare plot
        self.results.rolling(window=window, center=True).mean().plot(ax=ax, **kwargs)

        ax.set(title=f'Localizations per Frame\n (window={window})',
               xlabel = 'frame',
               ylabel = self.results.name
               )

        return ax

    def hist(self, ax=None, fit=True, bins='auto', **kwargs):
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing hist(results).

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which to show the image
        bins : float
            Bin specifications (passed to matplotlib.hist).
        fit: Bool
            Flag indicating if distributions fit are shown.
        kwargs : dict
            Other parameters passed to :func:`matplotlib.pyplot.hist`.

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        ax.hist(self.results.values, bins=bins, **dict(dict(density=True, log=False), **kwargs))
        ax.set(title = 'Localizations per Frame',
               xlabel = self.results.name,
               ylabel = 'PDF'
               )

        if fit:
            if isinstance(self.distribution_statistics, _DistributionFits):
                self.distribution_statistics.plot(ax=ax)
            else:
                self.fit_distributions()
                self.distribution_statistics.plot(ax=ax)

        return ax

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
    analyis_class : LocalizationPrecision
        The analysis class with result data to fit.
    loc_property : str
        The LocData property for which to fit an appropriate distribution
    distribution : str, scipy.stats.distribution
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
        distribution : str, scipy.stats.distribution
            Distribution model to fit.
        kwargs : dict
            Other parameters are passed to the `scipy.stat.distribution.fit()` function.
        """
        self.distribution = distribution

        loc, scale = self.distribution.fit(self.analysis_class.results.values, **kwargs)
        self.parameters.extend([self.loc_property + '_center', self.loc_property + '_sigma'])
        setattr(self, self.loc_property + '_center', loc)
        setattr(self, self.loc_property + '_sigma', scale)


    def plot(self, ax=None, **kwargs):
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the probability distribution functions of fitted
        results.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which to show the image.
        kwargs : dict
            Other parameters passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.distribution is None:
            return ax

        # plot fit curve
        _center, _sigma = self.parameter_dict().values()

        x_values = np.linspace(self.distribution.ppf(0.001, loc=_center, scale=_sigma),
                               self.distribution.ppf(0.999, loc=_center, scale=_sigma), 100)
        ax.plot(x_values, self.distribution.pdf(x_values, loc=_center, scale=_sigma), 'r-',
                **dict(dict(lw=3, alpha=0.6, label=str(self.distribution.name) + ' pdf'), **kwargs))
        return ax

    def parameter_dict(self):
        """ Dictionary of fitted parameters. """
        if self.parameters is None:
            return None
        else:
            return {k: self.__dict__[k] for k in self.parameters}