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
from surepy.analysis.analysis_base import _Analysis, _update_meta


##### The algorithms

def _localization_precision(locdata, radius=50):
    # group localizations
    grouped = locdata.data.groupby('Frame')

    # find nearest neighbors
    min = locdata.data['Frame'].unique().min()
    max = locdata.data['Frame'].unique().max()

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
                        df = df.rename(columns={'Position_x': 'Position_delta_x',
                                                'Position_y': 'Position_delta_y',
                                                'Position_z': 'Position_delta_z'})
                        df = df.assign(Position_distance=min_distance)
                        df = df.assign(Frame=i)
                        results = results.append(df)
        except KeyError:
            pass

    results.reset_index(inplace=True, drop=True)
    return (results)


##### The specific analysis classes


class Localization_precision(_Analysis):
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
        A counter for counting instantiations.
    locdata : LocData object
        Localization data.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : numpy array or pandas DataFrame
        Computed results.
    distribution_statistics : Distribution_fits object
        Distribution parameters derived from MLE fitting of results.
    """

    def __init__(self, locdata=None, meta=None, radius=50):
        super().__init__(locdata=locdata, meta=meta, radius=radius)

    def compute(self):
        """ Run the computation. """
        data = self.locdata
        self.results = _localization_precision(locdata=data, **self.parameter)
        return self

    def fit_distributions(self, loc_property=None):
        """
        Fit probability density functions to the distributions of `loc_property` values in the results
        using MLE (scipy.stats).

        Parameters
        ----------
        loc_property : LocData property
            The property for which to fit an appropriate distribution; if None all plots are shown.
        """
        self.distribution_statistics = Distribution_fits(self)
        if loc_property is None:
            for prop in ['Position_delta_x', 'Position_delta_y', 'Position_delta_z', 'Position_distance']:
                if prop in self.results.columns:
                    self.distribution_statistics.fit_distribution(loc_property=prop)
        else:
            self.distribution_statistics.fit_distribution(loc_property=loc_property)

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
        kwargs : dict
            Other parameters passed to matplotlib.pyplot.plot().
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # prepare plot
        self.results.rolling(window=window, center=True).mean().plot(ax=ax,
                                                                     x='Frame',
                                                                     y=loc_property,
                                                                     legend=False,
                                                                     ** kwargs)
        ax.set(title=f'Localization Precision (window={window})',
               xlabel='Frame',
               ylabel=loc_property
               )

        # show figure
        if show:
            plt.show()


    def hist(self, ax=None, show=True, loc_property='Position_distance', bins='auto', fit=True, **kwargs):
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
            self.distribution_statistics = Distribution_fits(analysis_class=self)
            self.distribution_statistics.fit_distribution(loc_property=loc_property)
            self.distribution_statistics.plot_distribution_fit(ax=ax, show=False, loc_property=loc_property)

        # show figure
        if show:
            plt.show()


#### Auxiliary functions and classes

class Pairwise_distance_distribution_2d(stats.rv_continuous):
    '''
    Define continuous distribution class (inheriting from scipy.stats.rv_continuous) for fitting the distribution
    of Position_distances (referred to as pairwise displacement distribution in [1]_)

    References
    ----------
    .. [1] Endesfelder, U., Malkusch, S., Fricke, F., and Heilemann, M. (2014) A simple method to estimate the average
       localization precision of a single-molecule localization microscopy experiment.
       Histochemistry and cell biology 141, 629–638

    '''

    def _pdf(self, x, sigma):
        return x / (2 * sigma ** 2) * np.exp(- x ** 2 / (4 * sigma ** 2))


class Distribution_fits:
    """
    Handle for distribution fits.

    This class is typically instantiated by Localization_precision methods.
    It holds the statistical parameters derived by fitting the result distributions using MLE (scipy.stats).

    Parameters
    ----------
    analyis_class : Localization_precision object
        The analysis class with result data to fit.

    Attributes
    ----------
    analyis_class : Localization_precision object
        The analysis class with result data to fit.
    pairwise_2D : Pairwise_distance_distribution_2d
        Continuous distribution function used to fit Position_distances
    Position_delta_x_center : float
        Center of the normal distribution fitted to Position_delta_x.
    Position_delta_x_sigma : float
        Sigma of the normal distribution fitted to Position_delta_x.
    Position_delta_y_center : float
        Center of the normal distribution fitted to Position_delta_y.
    Position_delta_y_sigma : float
        Sigma of the normal distribution fitted to Position_delta_y.
    Position_delta_z_center : float
        Center of the normal distribution fitted to Position_delta_z.
    Position_delta_z_sigma : float
        Sigma of the normal distribution fitted to Position_delta_z.
    Position_distance_sigma : float
        Sigma of the pairwise distance distribution from fitting Position_distance.
    """

    def __init__(self, analysis_class):
        self.analysis_class = analysis_class

        # continuous distributions
        self.pairwise_2D = Pairwise_distance_distribution_2d(name='pairwise', a=0.)
        # todo: 3D

        # fitted parameters for distributions
        self.Position_delta_x_center = None
        self.Position_delta_x_sigma = None
        self.Position_delta_y_center = None
        self.Position_delta_y_sigma = None
        self.Position_delta_z_center = None
        self.Position_delta_z_sigma = None
        self.Position_distance_sigma = None

    def fit_distribution(self, loc_property='Position_distance'):
        '''
        Fit distributions of results using a MLE fit (scipy.stats) and provide fit results.
        '''

        if 'Position_delta_' in loc_property:
            # MLE fit of distribution on data
            loc, scale = stats.norm.fit(self.analysis_class.results[loc_property].values)
            setattr(self, loc_property + '_center', loc)
            setattr(self, loc_property + '_sigma', scale)

        elif loc_property == 'Position_distance':
            # MLE fit of distribution on data with fixed loc and scale
            sigma, loc, scale = self.pairwise_2D.fit(self.analysis_class.results[loc_property].values, floc=0, fscale=1)
            self.Position_distance_sigma = sigma

    def plot_distribution_fit(self, ax=None, show=True, loc_property='Position_distance', **kwargs):
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
        kwargs : dict
            parameters passed to matplotlib.pyplot.plot().
        """

        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # plot fit curve
        if 'Position_delta_' in loc_property:
            _center = getattr(self, loc_property + '_center')
            _sigma = getattr(self, loc_property + '_sigma')

            x_values = np.linspace(stats.norm.ppf(0.01, loc=_center, scale=_sigma),
                                   stats.norm.ppf(0.99, loc=_center, scale=_sigma), 100)
            ax.plot(x_values, stats.norm.pdf(x_values, loc=_center, scale=_sigma), 'r-', lw=3, alpha=0.6,
                    label='fitted pdf', **kwargs)

        elif loc_property == 'Position_distance':
            _sigma = self.Position_distance_sigma
            x_values = np.linspace(self.pairwise_2D.ppf(0.01, sigma=_sigma),
                                   self.pairwise_2D.ppf(0.99, sigma=_sigma), 100)
            ax.plot(x_values, self.pairwise_2D.pdf(x_values, sigma=_sigma), 'r-', lw=3, alpha=0.6,
                    label='fitted pdf', **kwargs)

        if show:
            plt.show()

    def __str__(self):
        statistic_attributes = ['Position_delta_x_center', 'Position_delta_x_sigma',
                             'Position_delta_y_center', 'Position_delta_y_sigma',
                             'Position_delta_z_center', 'Position_delta_z_sigma',
                             'Position_distance_sigma']
        statistics_list = ''
        for key in statistic_attributes:
            if getattr(self, key, None) is not None:
                statistics_list += f'{key}: {getattr(self, key, None)}\n'
        return (statistics_list)
