"""

Nearest-neighbor distance distribution analysis

Nearest-neighbor distance distributions provide information about deviations from a spatial homogeneous Poisson process
(i.e. complete spatial randomness, CSR).
Point-event distances are given by the distance between a random point (not being an event) and the nearest event.
The point-event distance distribution is estimated from a number of random sample points and plotted in comparison to
the analytical function for equal localization density.

For a homogeneous 2D Poisson process with intensity :math:`\\rho` (expected number of points per unit area) the distance
from a randomly chosen event to the nearest other event (nearest-neighbor distance) is distributed according to the
following probability density (pdf) or cumulative density function (cdf) [1]_:

.. math::

   pdf(w) &= 2 \\rho \\pi w \\ exp(- \\rho \\pi w^2)

   cdf(w) &= 1 - exp (- \\rho \\pi w^2)


The same distribution holds for point-event distances if events are distributed as a homogeneous Poisson process with
intensity :math:`\\rho`.

References
----------
.. [1] Philip M. Dixon, Nearest Neighbor Methods,
   Department of Statistics, Iowa State University,
   20 December 2001

"""
import logging
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from locan.analysis.analysis_base import _Analysis
from locan.configuration import N_JOBS

__all__: list[str] = ["NearestNeighborDistances"]

logger = logging.getLogger(__name__)


# The algorithms


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
    a = (3 / 4 / np.pi / density) ** (1 / 3)
    return 3 / a * (x / a) ** 2 * np.exp(-((x / a) ** 3))


def _nearest_neighbor_distances(points, k=1, other_points=None):
    if other_points is None:
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=N_JOBS).fit(
            points
        )
        distances, indices = nn.kneighbors()
    else:
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=N_JOBS).fit(
            other_points
        )
        distances, indices = nn.kneighbors(points)

    return pd.DataFrame(
        {"nn_distance": distances[..., k - 1], "nn_index": indices[..., k - 1]}
    )


# The specific analysis classes


class NearestNeighborDistances(_Analysis):
    """
    Compute the k-nearest-neighbor distances within data or between data and other_data.

    The algorithm relies on sklearn.neighbors.NearestNeighbors.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    k : int
        Compute the kth nearest neighbor.


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
    distribution_statistics : Distribution_stats, None
        Distribution parameters derived from MLE fitting of results.
    """

    count = 0

    def __init__(self, meta=None, k=1):
        super().__init__(meta=meta, k=k)
        self.dimension = None
        self.localization_density = None
        self.results = None
        self.distribution_statistics = None

    def compute(self, locdata, other_locdata=None) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
           Localization data.
        other_locdata : LocData
            Other localization data from which nearest neighbors are taken.

        Returns
        -------
        Self
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        self.dimension = locdata.dimension
        # setting the localization density of locdata
        if other_locdata is None:
            self.localization_density = locdata.properties["localization_density_bb"]
        else:
            if other_locdata.dimension != self.dimension:
                raise TypeError(
                    "Dimensions for locdata and other_locdata must be identical."
                )
            self.localization_density = other_locdata.properties[
                "localization_density_bb"
            ]

        points = locdata.coordinates
        if other_locdata is None:
            other_points = None
        else:
            other_points = other_locdata.coordinates

        self.results = _nearest_neighbor_distances(
            points=points, **self.parameter, other_points=other_points
        )
        return self

    def fit_distributions(self, with_constraints=True):
        """
        Fit probability density functions to the distributions of `loc_property` values in the results
        using MLE (scipy.stats).

        If with_constraints is true we put the following constraints on the fit procedure:
        If distribution is expon then floc=np.min(self.analysis_class.results[self.loc_property].values).

        Parameters
        ----------
        distribution : str, scipy.stats.distribution
            Distribution model to fit.
        with_constraints : bool
            Flag to use predefined constraints on fit parameters.
        """
        if self:
            self.distribution_statistics = _DistributionFits(self)
            self.distribution_statistics.fit(with_constraints=with_constraints)
        else:
            logger.warning("No results available to fit.")

    def hist(self, ax=None, bins="auto", density=True, fit=False, **kwargs):
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing hist(results).

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which to show the image.
        bins : int, list, 'auto'
            Bin specification as used in :func:`matplotlib.hist`
        density : bool
            Flag for normalization as used in matplotlib.hist. True returns probability density function; None returns
            counts.
        fit : bool
            Flag indicating to fit pdf of nearest-neighbor distances under complete spatial randomness.
        kwargs : dict
            Other parameters passed to :func:`matplotlib.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        values, bin_values, patches = ax.hist(
            self.results["nn_distance"], bins=bins, density=density, label="data"
        )
        x_data = (bin_values[:-1] + bin_values[1:]) / 2

        if self.dimension == 2:
            ax.plot(
                x_data,
                pdf_nnDistances_csr_2D(x_data, self.localization_density),
                "r-",
                label="CSR",
                **kwargs,
            )
        elif self.dimension == 3:
            ax.plot(
                x_data,
                pdf_nnDistances_csr_3D(x_data, self.localization_density),
                "r-",
                label="CSR",
                **kwargs,
            )
        else:
            logger.warning(
                f"No analytic probability density function for {self.dimension} dimensions available."
            )

        # fit distributions:
        if fit:
            if isinstance(self.distribution_statistics, _DistributionFits):
                self.distribution_statistics.plot(ax=ax)
            else:
                self.fit_distributions()
                self.distribution_statistics.plot(ax=ax)

        ax.set(
            title="k-Nearest Neigbor Distances\n"
            + " (k = "
            + str(self.parameter["k"])
            + ")",
            xlabel="distance (nm)",
            ylabel="pdf" if density else "counts",
        )
        ax.legend(loc="best")

        return ax


# Auxiliary functions and classes


class NNDistances_csr_2d(stats.rv_continuous):
    """
    Continuous distribution function for nearest-neighbor distances of points distributed in 2D
    under complete spatial randomness.

    Parameters
    ----------
    density : float
        Shape parameter `density`, being the density of points.
    """

    def _pdf(self, x, density):
        return 2 * density * np.pi * x * np.exp(-density * np.pi * x**2)


class NNDistances_csr_3d(stats.rv_continuous):
    """
    Continuous distribution function for nearest-neighbor distances of points distributed in 3D
    under complete spatial randomness.

    Parameters
    ----------
    density : float
        Shape parameter `density`, being the density of points.
    """

    def _pdf(self, x, density):
        a = (3 / 4 / np.pi / density) ** (1 / 3)
        return 3 / a * (x / a) ** 2 * np.exp(-((x / a) ** 3))


class _DistributionFits:
    """
    Handle for distribution fits.

    This class is typically instantiated by LocalizationProperty methods.
    It holds the statistical parameters derived by fitting the result distributions using MLE (scipy.stats).
    Statistical parameters are defined as described in
    :ref:(https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html)

    Parameters
    ----------
    analyis_class : LocalizationPrecision
        The analysis class with result data to fit.

    Attributes
    ----------
    analyis_class : LocalizationPrecision
        The analysis class with result data to fit.
    loc_property : str
        The LocData property for which to fit an appropriate distribution
    distribution : str, scipy.stats.distribution
        Distribution model to fit.
    parameters : list of str
        Free parameters in `distribution`.
    """

    def __init__(self, analysis_class):
        self.analysis_class = analysis_class
        self.loc_property = "nn_distance"
        self.distribution = None
        self.parameters = []

    def fit(self, with_constraints=True, **kwargs):
        """
        Fit model function to analysis_class.results.

        If with_constraints is true (default) we put the following constraints on the fit procedure:
        loc=0, scale=1

        Parameters
        ----------
        distribution : str, scipy.stats.distribution
            Distribution model to fit.
        with_constraints : bool
            Flag to use predefined constraints on fit parameters.
        kwargs : dict
            Other parameters passed to the `distribution.fit()` method.
        """
        if self.analysis_class.dimension == 2:
            self.distribution = NNDistances_csr_2d(name="NNDistances_csr_2d", a=0.0)
        elif self.analysis_class.dimension == 3:
            self.distribution = NNDistances_csr_3d(name="NNDistances_csr_3d", a=0.0)
        else:
            logger.warning(
                f"No fit model for {self.analysis_class.dimension} dimensions available."
            )
            return

        self.parameters = [name.strip() for name in self.distribution.shapes.split(",")]
        self.parameters += ["loc", "scale"]

        if with_constraints:
            kwargs_ = dict(dict(floc=0, fscale=1), **kwargs)
        else:
            kwargs_ = kwargs

        fit_results = self.distribution.fit(
            data=self.analysis_class.results[self.loc_property].values, **kwargs_
        )
        for parameter, result in zip(self.parameters, fit_results):
            setattr(self, parameter, result)

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
        x_values = np.linspace(
            self.distribution.ppf(0.001, **self.parameter_dict()),
            self.distribution.ppf(0.999, **self.parameter_dict()),
            100,
        )
        ax.plot(
            x_values,
            self.distribution.pdf(x_values, **self.parameter_dict()),
            "r-",
            **dict(
                dict(lw=3, alpha=0.6, label=str(self.distribution.name) + " pdf"),
                **kwargs,
            ),
        )
        return ax

    def parameter_dict(self):
        """Dictionary of fitted parameters."""
        return {k: self.__dict__[k] for k in self.parameters}
