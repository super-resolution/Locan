"""
This module provides methods for nearest-neighbor analysis.
"""
# todo: add fit

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import matplotlib.pyplot as plt

from surepy.analysis.analysis_base import _Analysis, _list_parameters
from surepy.constants import N_JOBS


#### The algorithms

def pdf_nnDistances_csr_2D(x, density):
    """
    Probability density function for nearest-neighbor distances of points distributed in 2D with complete spatial
    randomness.

    Parameters
    ----------
    x : float
        distance
    density : float
        density of points

    Returns
    -------
    float
        Probability density function pdf(x).
    """
    return 2 * density * np.pi * x * np.exp(-density * np.pi * x**2)


def pdf_nnDistances_csr_3D(x, density):
    """
    Probability density function for nearest-neighbor distances of points distributed in 3D with complete spatial
    randomness.

    Parameters
    ----------
    x : float
        distance
    density : float
        density of points

    Returns
    -------
    float
        Probability density function pdf(x).
    """
    a = (3 / 4 / np.pi / density)**(1/3)
    return 3 / a * (x/a)**2 * np.exp(-(x/a)**3)


def _nearest_neighbor_distances(points, k=1, other_points=None):

    if other_points is None:
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=N_JOBS).fit(points)
        distances, indices = nn.kneighbors()
    else:
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=N_JOBS).fit(other_points)
        distances, indices = nn.kneighbors(points)

    return pd.DataFrame({'nn_distance': distances[...,k-1], 'nn_index': indices[...,k-1]})


# The specific analysis classes

class Nearest_neighbor_distances(_Analysis):
    '''
    Compute the k-nearest-neighbor distances within data or between data and other_data.

    The algorithm relies on sklearn.neighbors.NearestNeighbors.

    Parameters
    ----------
    locdata : LocData object
        Localization data.
    k : int
        Compute the kth nearest neighbor.
    other_locdata : LocData object
        Other localization data from which nearest neighbors are taken.

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
    distribution_statistics : Distribution_stats object, None
        Distribution parameters derived from MLE fitting of results.
    '''
    count = 0

    def __init__(self, locdata=None, meta=None, k=1, other_locdata=None):
        super().__init__(locdata=locdata, meta=meta, k=k, other_locdata=other_locdata)
        self.distribution_statistics = None

        #setting the localization density of locdata
        if self.parameter['other_locdata'] is None:
            self.localization_density = self.locdata.properties['Localization_density_bb']
        else:
            self.localization_density = self.parameter['other_locdata'].properties['Localization_density_bb']

    def compute(self):
        points = self.locdata.coordinates

        # turn other_locdata into other_points
        new_parameter = {key: self.parameter[key] for key in self.parameter if key is not 'other_locdata'}
        if self.parameter['other_locdata'] is not None:
            other_points = self.parameter['other_locdata'].coordinates
        else:
            other_points = None

        self.results = _nearest_neighbor_distances(points=points,
                                                   **new_parameter, other_points=other_points)
        return self


    def fit_distributions(self, with_constraints=True):
        """
        Fit probability density functions to the distributions of `loc_property` values in the results
        using MLE (scipy.stats).

        If with_constraints is true we put the following constraints on the fit procedure:
        If distribution is expon then floc=np.min(self.analysis_class.results[self.loc_property].values).

        Parameters
        ----------
        distribution : str or scipy.stats distribution object
            Distribution model to fit.
        with_constraints : bool
            Flag to use predefined constraints on fit parameters.
        """
        self.distribution_statistics = _DistributionFits(self)
        self.distribution_statistics.fit(with_constraints=with_constraints)


    def hist(self, ax=None, show=True, bins='auto', density=True, fit=False, **kwargs):
        """
        Provide histogram as matplotlib axes object showing hist(results).

        Parameters
        ----------
        bins : int, list or 'auto'
            Bin specification as used in matplotlib.hist
        density : bool
            Flag for normalization as used in matplotlib.hist. True returns probability density function; None returns
            counts.
        fit : bool
            Flag indicating to fit pdf of nearest-neighbor distances under complete spatial randomness.

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to matplotlib.plot().
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        values, bin_values, patches = ax.hist(self.results['nn_distance'], bins=bins, density=density, label = 'data')
        x_data = (bin_values[:-1] + bin_values[1:]) / 2

        ax.plot(x_data, pdf_nnDistances_csr_2D(x_data, self.localization_density), 'r-', label='CSR', **kwargs)

        # fit distributions:
        if fit:
            if isinstance(self.distribution_statistics, _DistributionFits):
                self.distribution_statistics.plot(ax=ax, show=False)
            else:
                self.fit_distributions()
                self.distribution_statistics.plot(ax=ax, show=False)

        ax.set(title = 'k-Nearest Neigbor Distances'+' (k = ' + str(self.parameter['k'])+')',
               xlabel = 'distance (nm)',
               ylabel = 'pdf' if density else 'counts'
               )
        ax.legend(loc = 'best')

        # show figure
        if show:
            plt.show()

        return None


#### Auxiliary functions and classes

class NNDistances_csr_2d(stats.rv_continuous):
    """
    Define continuous distribution class (inheriting from scipy.stats.rv_continuous) for fitting the distribution
    of nearest-neighbor distances of points distributed in 2D with complete spatial randomness [1]_.

    Parameters
    ----------
    x : float
        distance
    density : float
        density of points

    References
    ----------
    .. [1] todo

    """

    def _pdf(self, x, density):
        return 2 * density * np.pi * x * np.exp(-density * np.pi * x**2)


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
    parameters : list of str
        Free parameters in `distribution`.
    """
    def __init__(self, analysis_class):
        self.analysis_class = analysis_class
        self.loc_property = 'nn_distance'
        self.distribution = None
        self.parameters = None

    def fit(self, with_constraints=True):
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
        """
        self.distribution = NNDistances_csr_2d()
        # todo: add 3D
        self.parameters = [name.strip() for name in self.distribution.shapes.split(',')]
        self.parameters += ['loc', 'scale']

        if with_constraints:
            fit_results = self.distribution.fit(data=self.analysis_class.results[self.loc_property].values,
                                                density=self.analysis_class.localization_density, floc=0, fscale=1)
            for parameter, result in zip(self.parameters, fit_results):
                setattr(self, parameter, result)
        else:
            fit_results = self.distribution.fit(data=self.analysis_class.results[self.loc_property].values,
                                                density=self.analysis_class.localization_density)
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
        x_values = np.linspace(self.distribution.ppf(0.001, **self.parameter_dict()),
                               self.distribution.ppf(0.999, **self.parameter_dict()), 100)
        ax.plot(x_values, self.distribution.pdf(x_values, **self.parameter_dict()), 'r-', lw=3, alpha=0.6,
                label=str(self.distribution) + ' pdf', **kwargs)
        if show:
            plt.show()

    def parameter_dict(self):
        """ Dictionary of fitted parameters. """
        if self.parameters is None:
            return None
        else:
            return {k: self.__dict__[k] for k in self.parameters}
