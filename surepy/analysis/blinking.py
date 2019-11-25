"""
Compute on- and off-periods from localization frames.

Assuming that the provided localizations are acquired from the same label, we analyze the times of recording as
provided by the `frame` property.

Note
----
The data is currently fitted to an exponential function. This is not correct and will be changed soon.

"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from surepy.analysis.analysis_base import _Analysis, _list_parameters
from surepy.data.locdata import LocData


__all__ = ['BlinkStatistics']


##### The algorithms

def _blink_statistics(locdata, memory=0, remove_heading_off_periods=True):
    """
    Estimate on and off times from the frame values provided.

    On and off-periods are determined from the sorted frame values.
    A series of frame values that constantly increase by one is considered a on-period.
    Each series of missing frame values between two given frame values is considered an off-period.

    Parameters
    ----------
    locdata : LocData or array-like
        Localization data or just the frame values of given localizations.
    memory : int
        The maximum number of intermittent frames without any localization
        that are still considered to belong to the same on-period.

    Returns
    -------
    dict with values being ndarrays
        'on_periods' and 'off_periods' in units of frame numbers.
    """
    if isinstance(locdata, LocData):
        frames = locdata.data.frame.values
    else:
        frames = locdata

    frames, counts = np.unique(frames, return_counts=True)

    # provide warning if duplicate frames are found. This should not be the case for appropriate localization clusters.
    if np.any(counts > 1):
        counts_larger_one = counts[counts > 1]
        warnings.warn(f'There are {sum(counts_larger_one) - len(counts_larger_one)} '
                      f'duplicated frames found that will be ignored.')

    # frames are counted from 0. We change this to start with 1 and insert 0 to get a 1 frame on period
    # for a localization in frame 0.
    frames = frames + 1
    frames = np.insert(frames, 0, 0)

    differences = np.diff(frames)
    indices = np.nonzero(differences > memory + 1)[0]
    groups = np.split(differences, indices)

    if groups[0].size == 0:
        groups = groups[1:]

    # on-times
    # the sum is taken to include memory>0.
    # one is added since a single localization is considered to be on for one frame.
    on_periods = np.array([np.sum(group[1:]) + 1 for group in groups])

    # off-times
    off_periods = np.array([group[0] - 1 for group in groups])
    if off_periods[0] == 0:
        off_periods = off_periods[1:]
    else:
        if remove_heading_off_periods:
            off_periods = off_periods[1:]

    return dict(on_periods=on_periods, off_periods=off_periods)


##### The specific analysis classes

class BlinkStatistics(_Analysis):
    """
    Estimate on and off times from the frame values provided.

    On and off-periods are determined from the sorted frame values.
    A series of frame values that constantly increase by one is considered a on-period.
    Each series of missing frame values between two given frame values is considered an off-period.

    Parameters
    ----------
    memory : int
        The maximum number of intermittent frames without any localization
        that are still considered to belong to the same on-period.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.

    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : tuple of ndarrays
        Two arrays with on- and off-periods in units of frame numbers.
    """
    count = 0

    def __init__(self, meta=None, memory=0, remove_heading_off_periods=True):
        super().__init__(meta, memory=memory, remove_heading_off_periods=remove_heading_off_periods)
        self.results = None
        self.distribution_statistics = {}

    def compute(self, locdata):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData or array-like
            Localization data or just the frame values of given localizations.

        Returns
        -------
        Analysis class
            Returns the Analysis class object (self).
        """
        self.results = _blink_statistics(locdata=locdata, **self.parameter)
        return self

    def fit_distributions(self, distribution=stats.expon, data_identifier=('on_periods', 'off_periods'),
                          with_constraints=True, **kwargs):
        """
        Fit probability density functions to the distributions of on- and off-periods in the results
        using MLE (scipy.stats).

        If with_constraints is true we put the following constraints on the fit procedure:
        If distribution is expon then floc=np.min(self.analysis_class.results[self.loc_property].values).

        Parameters
        ----------
        distribution : str or scipy.stats distribution object
            Distribution model to fit.
        data_identifier : str
            String to identify the data in `results` for which to fit an appropriate distribution, here
            'on_periods' or 'off_periods'. For True all are fitted.
        with_constraints : bool
            Flag to use predefined constraints on fit parameters.

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters are passed to the `scipy.stat.distribution.fit()` function.
        """
        if isinstance(data_identifier, (tuple, list)):
            data_identifier_ = data_identifier
        else:
            data_identifier_ = (data_identifier,)

        for data_id in data_identifier_:
            self.distribution_statistics[data_id] = _DistributionFits(self, data_identifier=data_id,
                                             distribution=distribution)
            self.distribution_statistics[data_id].fit(with_constraints=with_constraints, **kwargs)

    def hist(self, data_identifier='on_periods', ax=None, bins='auto', log=True, fit=True, **kwargs):
        """
        Provide histogram as matplotlib axes object showing hist(results).

        Parameters
        ----------
        data_identifier : str
            'on_periods' or 'off_periods'.
        ax : matplotlib axes
            The axes on which to show the image
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

        Returns
        -------
        matplotlib Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        ax.hist(self.results[data_identifier], bins=bins, density=True, log=log, **kwargs)
        ax.set(title = f'Distribution of {data_identifier}',
               xlabel = f'{data_identifier} (frames)',
               ylabel = 'PDF'
               )

        # fit distributions:
        if fit:
            if data_identifier in self.distribution_statistics and \
                    isinstance(self.distribution_statistics[data_identifier], _DistributionFits):
                self.distribution_statistics[data_identifier].plot(ax=ax)
            else:
                self.fit_distributions(data_identifier=data_identifier)
                self.distribution_statistics[data_identifier].plot(ax=ax)

        return ax


# todo: incorporate fitting of geometrical distribution instead of exponential.
class _DistributionFits:
    """
    Handle for distribution fits.

    This class is typically instantiated by specific Analysis methods.
    It holds the statistical parameters derived by fitting the result distributions using MLE (scipy.stats).
    Statistical parameters are defined as described in
    :ref:(https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html)

    Parameters
    ----------
    analyis_class : Analysis object
        The analysis class with result data to fit.
    distribution : str or scipy.stats distribution object
        Distribution model to fit.
    data_identifier : str
        String to identify the data in `results` for which to fit an appropriate distribution

    Attributes
    ----------
    analyis_class : Analysis object
        The analysis class with result data to fit.
    distribution : str or scipy.stats distribution object
        Distribution model to fit.
    data_identifier : str
        String to identify the data in `results` for which to fit an appropriate distribution
    parameters : list of string
        Distribution parameters.
    """
    def __init__(self, analysis_class, distribution, data_identifier):
        self.analysis_class = analysis_class
        self.distribution = distribution
        self.data_identifier = data_identifier
        self.parameters = []

    def __repr__(self):
        """ Return representation of the _DistributionFits class. """
        param_dict = dict(analysis_class=self.analysis_class.__class__.__name__,
                          distribution=self.distribution.__class__.__name__,
                          data_identifier=self.data_identifier)
        param_string = ''.join([f'{key}={val}, ' for key, val in param_dict.items()])
        param_string = param_string[:-2]
        return f'{self.__class__.__name__}({param_string})'

    def fit(self, with_constraints=True, **kwargs):
        """
        Fit scipy.stats.distribution to analysis_class.results[data_identifier].

        If with_constraints is true we put the following constraints on the fit procedure:
        If distribution is expon then floc=np.min(self.analysis_class.results[self.data_identifier].values).

        Parameters
        ----------
        with_constraints : bool
            Flag to use predefined constraints on fit parameters.

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters are passed to the `scipy.stat.distribution.fit()` function.
        """
        # set data
        if isinstance(self.analysis_class.results, pd.DataFrame):
            data = self.analysis_class.results[self.data_identifier].values
        else:
            data = self.analysis_class.results[self.data_identifier]

        # define parameter names
        for param in _list_parameters(self.distribution):
            self.parameters.append(self.data_identifier + '_' + param)

        # perform fit
        if with_constraints and self.distribution == stats.expon:
            # MLE fit of exponential distribution with constraints
            fit_results = stats.expon.fit(data, floc=np.min(data), **kwargs)
            for parameter, result in zip(self.parameters, fit_results):
                setattr(self, parameter, result)
        else:
            fit_results = self.distribution.fit(data, **kwargs)
            for parameter, result in zip(self.parameters, fit_results):
                setattr(self, parameter, result)

    def plot(self, ax=None, **kwargs):
        """
        Provide plot as matplotlib axes object showing the probability distribution functions of fitted results.

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image.

        Other Parameters
        ----------------
        kwargs : dict
            parameters passed to matplotlib.pyplot.plot().

        Returns
        -------
        matplotlib Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        # plot fit curve
        parameter = self.parameter_dict().values()
        x_values = np.linspace(self.distribution.ppf(1e-4, *parameter),
                               self.distribution.ppf(1 - 1e-4, *parameter), 100)
        ax.plot(x_values, self.distribution.pdf(x_values, *parameter), 'r-', lw=3, alpha=0.6,
                label=str(self.distribution) + ' pdf', **kwargs)

        return ax

    def parameter_dict(self):
        """ Dictionary of fitted parameters. """
        return {k: self.__dict__[k] for k in self.parameters}
