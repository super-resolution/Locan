"""

Compute localization precision from successive nearby localizations.

Localization precision is estimated from spatial variations of all localizations that appear in successive frames
within a specified search radius following [1]_.

References
----------
.. [1] Endesfelder, Ulrike, et al., A simple method to estimate the average localization precision of a single-molecule
   localization microscopy experiment. Histochemistry and Cell Biology 141.6 (2014): 629-638.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from scipy import stats

from surepy.constants import N_JOBS
from surepy.analysis.analysis_base import _Analysis, _list_parameters


##### The algorithms

def _localization_precision(locdata, radius=50):
    # group localizations
    grouped = locdata.data.groupby('frame')

    # find nearest neighbors
    min = locdata.data['frame'].unique().min()
    max = locdata.data['frame'].unique().max()

    results = pd.DataFrame()

    for i in range(min, max - 1):
        try:
            points = grouped.get_group(i)[locdata.coordinate_labels]
            other_points = grouped.get_group(i + 1)[locdata.coordinate_labels]

            # print(points)

            nn = NearestNeighbors(radius=radius, metric='euclidean', n_jobs=N_JOBS).fit(other_points)
            distances, indices = nn.radius_neighbors(points)

            if len(distances):
                for n, (dists, inds) in enumerate(zip(distances, indices)):
                    if len(dists):
                        min_distance = np.amin(dists)
                        min_position = np.argmin(dists)
                        min_index = inds[min_position]
                        difference = points.iloc[n] - other_points.iloc[min_index]

                        df = difference.to_frame().T
                        df = df.rename(columns={'position_x': 'position_delta_x',
                                                'position_y': 'position_delta_y',
                                                'position_z': 'position_delta_z'})
                        df = df.assign(position_distance=min_distance)
                        df = df.assign(frame=i)
                        results = results.append(df)
        except KeyError:
            pass

    results.reset_index(inplace=True, drop=True)
    return (results)


##### The specific analysis classes


class LocalizationPrecision(_Analysis):
    """
    Compute the localization precision from consecutive nearby localizations.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    radius : int or float
        Search radius for nearest-neighbor searches.

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
    distribution_statistics : Distribution_fits object, None
        Distribution parameters derived from MLE fitting of results.
    """
    def __init__(self, meta=None, radius=50):
        super().__init__(meta=meta, radius=radius)
        self.results = None
        self.distribution_statistics = None

    def compute(self, locdata):
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData object
            Localization data.

        Returns
        -------
        Analysis class
            Returns the Analysis class object (self).
        """
        self.results = _localization_precision(locdata=locdata, **self.parameter)
        return self

    def fit_distributions(self, loc_property=None, **kwargs):
        """
        Fit probability density functions to the distributions of `loc_property` values in the results
        using MLE (scipy.stats).

        Parameters
        ----------
        loc_property : LocData property
            The property for which to fit an appropriate distribution; if None all plots are shown.

        Other Parameters
        ----------------
        kwargs : dict
            Parameters passed to the `distribution.fit()` method.
        """
        self.distribution_statistics = _DistributionFits(self)
        if loc_property is None:
            for prop in ['position_delta_x', 'position_delta_y', 'position_delta_z', 'position_distance']:
                if prop in self.results.columns:
                    self.distribution_statistics.fit(loc_property=prop, **kwargs)
        else:
            self.distribution_statistics.fit(loc_property=loc_property, **kwargs)

    def plot(self, ax=None, show=True, loc_property=None, window=1, **kwargs):
        """
        Provide plot as matplotlib axes object showing the running average of results over window size.

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image
        show : bool
            Flag indicating if plt.show() is active.
        loc_property : LocData property
            The property for which to plot localization precision; if None all plots are shown.
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
        self.results.rolling(window=window, center=True).mean().plot(ax=ax,
                                                                     x='frame',
                                                                     y=loc_property,
                                                                     legend=False,
                                                                     ** kwargs)
        ax.set(title=f'Localization Precision\n (window={window})',
               xlabel='frame',
               ylabel=loc_property
               )

        # show figure
        if show:
            plt.show()


    def hist(self, ax=None, show=True, loc_property='position_distance', bins='auto', fit=True, **kwargs):
        """
        Provide histogram as matplotlib axes object showing the distributions of results.

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image
        show : bool
            Flag indicating if plt.show() is active.
        loc_property : LocData property
            The property for which to plot localization precision.
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

        # prepare plot
        ax.hist(self.results[loc_property].values, bins=bins, density=True, log=False, **kwargs)
        ax.set(title='Localization Precision',
               xlabel=loc_property,
               ylabel='PDF'
               )

        if fit:
            if isinstance(self.distribution_statistics, _DistributionFits):
                self.distribution_statistics.plot(ax=ax, show=False, loc_property=loc_property)
            else:
                self.fit_distributions()
                self.distribution_statistics.plot(ax=ax, show=False, loc_property=loc_property)

        # show figure
        if show:
            plt.show()


#### Auxiliary functions and classes

class Pairwise_distance_distribution_2d(stats.rv_continuous):
    '''
    A random variable describing the distribution
    of Position_distances (referred to as pairwise displacement distribution in [1]_)

    The continuous distribution class inherits from scipy.stats.rv_continuous a set of methods and is defined by
    overriding the _pdf method.

    Parameters
    ----------
    shapes : float
        Shape parameter `sigma`.

    References
    ----------
    .. [1] Endesfelder, U., Malkusch, S., Fricke, F., and Heilemann, M. (2014) A simple method to estimate the average
       localization precision of a single-molecule localization microscopy experiment.
       Histochemistry and cell biology 141, 629–638

    '''

    def _pdf(self, x, sigma):
        return x / (2 * sigma ** 2) * np.exp(- x ** 2 / (4 * sigma ** 2))


class _DistributionFits:
    """
    Handle for distribution fits.

    This class is typically instantiated by LocalizationPrecision methods.
    It holds the statistical parameters derived by fitting the result distributions using MLE (scipy.stats).

    Parameters
    ----------
    analyis_class : LocalizationPrecision object
        The analysis class with result data to fit.

    Attributes
    ----------
    analyis_class : LocalizationPrecision object
        The analysis class with result data to fit.
    pairwise_distribution : Pairwise_distance_distribution_2d
        Continuous distribution function used to fit Position_distances
    parameters : list of string
        Distribution parameters.

    Notes
    -----
    Attributes for fit parameter are generated dynamically, named as loc_property + distribution parameters and listed
    in parameters.
    """

    def __init__(self, analysis_class):
        self.analysis_class = analysis_class
        self.distribution = None
        self._dist_parameters = None
        self.parameters = []

        # continuous distributions
        self.pairwise_distribution = Pairwise_distance_distribution_2d(name='pairwise', a=0.)
        # todo: 3D

    def fit(self, loc_property='position_distance', **kwargs):
        '''
        Fit distributions of results using a MLE fit (scipy.stats) and provide fit results.

        Parameters
        ----------
        loc_property : LocData property
            The property for which to fit an appropriate distribution

        Other Parameters
        ----------------
        kwargs : dict
            Parameters passed to the `distribution.fit()` method.
        '''
        # prepare parameters
        if 'position_delta_' in loc_property:
            self.distribution = stats.norm
            self._dist_parameters = [(loc_property + '_' + param) for param in ['loc', 'scale']]
        elif loc_property == 'position_distance':
            self.distribution = self.pairwise_distribution
            self._dist_parameters = [(loc_property + '_' + param) for param in ['sigma', 'loc', 'scale']]
        else:
            raise TypeError('Unknown localization property.')

        for param in self._dist_parameters:
            if param not in self.parameters:
                self.parameters.append(param)

        # MLE fit of distribution on data
        if 'position_delta_' in loc_property:
            fit_results = self.distribution.fit(self.analysis_class.results[loc_property].values, **kwargs)
        elif loc_property == 'position_distance':
            fit_results = self.distribution.fit(self.analysis_class.results[loc_property].values,
                                                floc=0, fscale=1, **kwargs)
        else:
            raise TypeError('Unknown localization property.')

        for parameter, result in zip(self._dist_parameters, fit_results):
            setattr(self, parameter, result)

    def plot(self, ax=None, show=True, loc_property='position_distance', **kwargs):
        """
        Provide plot as matplotlib axes object showing the probability distribution functions of fitted results.

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image.
        show : bool
            Flag indicating if plt.show() is active.
        loc_property : LocData property
            The property for which to plot the distribution fit.

        Other Parameters
        ----------------
        kwargs : dict
            parameters passed to matplotlib.pyplot.plot().
        """

        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # plot fit curve
        if 'position_delta_' in loc_property:
            _loc = getattr(self, loc_property + '_loc')
            _scale = getattr(self, loc_property + '_scale')

            x_values = np.linspace(stats.norm.ppf(0.01, loc=_loc, scale=_scale),
                                   stats.norm.ppf(0.99, loc=_loc, scale=_scale), 100)
            ax.plot(x_values, stats.norm.pdf(x_values, loc=_loc, scale=_scale), 'r-', lw=3, alpha=0.6,
                    label='fitted pdf', **kwargs)

        elif loc_property == 'position_distance':
            _sigma = self.position_distance_sigma
            x_values = np.linspace(self.pairwise_distribution.ppf(0.01, sigma=_sigma),
                                   self.pairwise_distribution.ppf(0.99, sigma=_sigma), 100)
            ax.plot(x_values, self.pairwise_distribution.pdf(x_values, sigma=_sigma), 'r-', lw=3, alpha=0.6,
                    label='fitted pdf', **kwargs)

        if show:
            plt.show()

    def parameter_dict(self):
        """ Dictionary of fitted parameters. """
        return {k: self.__dict__[k] for k in self.parameters}
