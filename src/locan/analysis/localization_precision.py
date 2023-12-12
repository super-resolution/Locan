"""
Compute localization precision from successive nearby localizations.

Localization precision is estimated from spatial variations of all
localizations that appear in successive frames
within a specified search radius [1]_.

Localization pair distance distributions are fitted according to the
probability density functions in [2]_.

The estimated sigmas describe the standard deviation for pair distances.
Localization precision is often defined as the standard deviation for
localization distances from the center position.
With that definition, the localization precision is equal to sigma / sqrt(2).

References
----------
.. [1] Endesfelder, Ulrike, et al.,
   A simple method to estimate the average localization precision of a
   single-molecule localization microscopy experiment.
   Histochemistry and Cell Biology 141.6 (2014): 629-638.

.. [2] L. Stirling Churchman, Henrik Flyvbjerg, James A. Spudich,
   A Non-Gaussian Distribution Quantifies Distances Measured with Fluorescence
   Localization Techniques.
   Biophysical Journal 90 (2), 2006, 668-671,
   doi.org/10.1529/biophysj.105.065599.
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    import matplotlib as mpl

    from locan.data.locdata import LocData

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis
from locan.configuration import N_JOBS, TQDM_DISABLE, TQDM_LEAVE

__all__: list[str] = ["LocalizationPrecision"]

logger = logging.getLogger(__name__)


# The algorithms


def _localization_precision(locdata: LocData, radius: int | float = 50) -> pd.DataFrame:
    # group localizations
    grouped = locdata.data.groupby("frame")

    # find nearest neighbors
    min = locdata.data["frame"].unique().min()
    max = locdata.data["frame"].unique().max()

    results = pd.DataFrame()

    for i in tqdm(
        range(min, max - 1),
        desc="Processed frames:",
        leave=TQDM_LEAVE,
        disable=TQDM_DISABLE,
    ):
        try:
            points = grouped.get_group(i)[locdata.coordinate_keys]
            other_points = grouped.get_group(i + 1)[locdata.coordinate_keys]

            nn = NearestNeighbors(radius=radius, metric="euclidean", n_jobs=N_JOBS).fit(
                other_points
            )
            distances, indices = nn.radius_neighbors(points)

            if len(distances):
                for n, (dists, inds) in enumerate(zip(distances, indices)):
                    if len(dists):
                        min_distance = np.amin(dists)
                        min_position = np.argmin(dists)
                        min_index = inds[min_position]
                        difference = points.iloc[n] - other_points.iloc[min_index]

                        df = difference.to_frame().T
                        df = df.rename(
                            columns={
                                "position_x": "position_delta_x",
                                "position_y": "position_delta_y",
                                "position_z": "position_delta_z",
                            }
                        )
                        df = df.assign(position_distance=min_distance)
                        df = df.assign(frame=i)
                        results = pd.concat([results, df])
        except KeyError:
            pass

    results.reset_index(inplace=True, drop=True)
    return results


# The specific analysis classes


class LocalizationPrecision(_Analysis):
    """
    Compute the localization precision from consecutive nearby localizations.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata | None
        Metadata about the current analysis routine.
    radius : int | float
        Search radius for nearest-neighbor searches.

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : pandas.DataFrame
        Computed results.
    distribution_statistics : Distribution_fits | None
        Distribution parameters derived from MLE fitting of results.
    """

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        radius: int | float = 50,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.results = None
        self.distribution_statistics: _DistributionFits | None = None

    def compute(self, locdata: LocData) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata
            Localization data.

        Returns
        -------
        Self
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        self.results = _localization_precision(locdata=locdata, **self.parameter)
        if self.results.empty:
            logger.warning("No successive localizations were found.")

        return self

    def fit_distributions(self, loc_property: str | None = None, **kwargs: Any) -> None:
        """
        Fit probability density functions to the distributions of
        `loc_property` values in the results using MLE (scipy.stats).

        Parameters
        ----------
        loc_property : str
            The LocData property for which to fit an appropriate distribution;
            if None all plots are shown.
        kwargs
            Other parameters passed to the `distribution.fit()` method.
        """
        if self.results is None:
            logger.warning("No results available to be fitted.")
            return

        self.distribution_statistics = _DistributionFits(self)

        if loc_property is None:
            for prop in [
                "position_delta_x",
                "position_delta_y",
                "position_delta_z",
                "position_distance",
            ]:
                if prop in self.results.columns:
                    self.distribution_statistics.fit(loc_property=prop, **kwargs)
        else:
            self.distribution_statistics.fit(loc_property=loc_property, **kwargs)

    def plot(
        self,
        ax: mpl.axes.Axes | None = None,
        loc_property: str | list[str] | None = None,
        window: int = 1,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        running average of results over window size.

        Parameters
        ----------
        ax
            The axes on which to show the image
        loc_property
            The property for which to plot localization precision;
            if None all plots are shown.
        window
            Window for running average that is applied before plotting.
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
            return ax

        # prepare plot
        self.results.rolling(window=window, center=True).mean().plot(
            ax=ax, x="frame", y=loc_property, **dict(dict(legend=False), **kwargs)
        )
        ax.set(
            title=f"Localization Precision\n (window={window})",
            xlabel="frame",
            ylabel=loc_property,
        )

        return ax

    def hist(
        self,
        ax: mpl.axes.Axes | None = None,
        loc_property: str = "position_distance",
        bins: int | Sequence[int | float] | str = "auto",
        fit: bool = True,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing the
        distributions of results.

        Parameters
        ----------
        ax
            The axes on which to show the image
        loc_property
            The property for which to plot localization precision.
        bins
            Bin specifications (passed to :func:`matplotlib.hist`).
        fit
            Flag indicating if distributions fit are shown.
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.his`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
            return ax

        # prepare plot
        ax.hist(
            self.results[loc_property].values,
            bins=bins,
            **dict(dict(density=True, log=False), **kwargs),
        )
        ax.set(title="Localization Precision", xlabel=loc_property, ylabel="PDF")

        if fit:
            if self.distribution_statistics is None:
                self.fit_distributions()

            assert (  # type narrowing # noqa: S101
                self.distribution_statistics is not None
            )
            self.distribution_statistics.plot(ax=ax, loc_property=loc_property)

        return ax


# Auxiliary functions and classes


class PairwiseDistance1d(stats.rv_continuous):
    """
    A random variable describing the distribution
    of pair distances for two normal distributed point clouds in 1D.

    The continuous distribution class inherits from scipy.stats.rv_continuous
    a set of methods and is defined by overriding the _pdf method.

    For theoretical background see [1]_.

    Parameters
    ----------
    x : npt.ArrayLike
        distance
    mu : npt.ArrayLike
        Distance between the point cloud center positions.
    sigma_1 : npt.ArrayLike
        Standard deviation for one point cloud.
    sigma_2 : npt.ArrayLike
        Standard deviation for the other point cloud.

    References
    ----------
    .. [1] L. Stirling Churchman, Henrik Flyvbjerg, James A. Spudich,
       A Non-Gaussian Distribution Quantifies Distances Measured with
       Fluorescence Localization Techniques.
       Biophysical Journal 90 (2), 2006, 668-671,
       doi.org/10.1529/biophysj.105.065599.
    """

    def _pdf(
        self,
        x: npt.ArrayLike,
        mu: npt.ArrayLike,
        sigma_1: npt.ArrayLike,
        sigma_2: npt.ArrayLike,
    ) -> npt.NDArray[np.float_]:
        x = np.asarray(x)
        mu = np.asarray(mu)
        sigma_1 = np.asarray(sigma_1)
        sigma_2 = np.asarray(sigma_2)

        sigma = np.sqrt(sigma_1**2 + sigma_2**2)
        return_value: npt.NDArray[np.float_] = (
            np.sqrt(2 / np.pi)
            / sigma
            * np.exp(-(mu**2 + x**2) / (2 * sigma**2))
            * np.cosh(x * mu / (sigma**2))
        )
        return return_value


class PairwiseDistance2d(stats.rv_continuous):
    """
    A random variable describing the distribution
    of pair distances for two normal distributed point clouds in 2D.

    The continuous distribution class inherits from scipy.stats.rv_continuous
    a set of methods and is defined by overriding the _pdf method.

    For theoretical background see [1]_.

    Parameters
    ----------
    x : npt.ArrayLike
        distance
    mu : npt.ArrayLike
        Distance between the point cloud center positions.
    sigma_1 : npt.ArrayLike
        Standard deviation for one point cloud.
    sigma_2 : npt.ArrayLike
        Standard deviation for the other point cloud.

    References
    ----------
    .. [1] L. Stirling Churchman, Henrik Flyvbjerg, James A. Spudich,
       A Non-Gaussian Distribution Quantifies Distances Measured with
       Fluorescence Localization Techniques.
       Biophysical Journal 90 (2), 2006, 668-671,
       doi.org/10.1529/biophysj.105.065599.
    """

    def _pdf(
        self,
        x: npt.ArrayLike,
        mu: npt.ArrayLike,
        sigma_1: npt.ArrayLike,
        sigma_2: npt.ArrayLike,
    ) -> npt.NDArray[np.float_]:
        x = np.asarray(x)
        mu = np.asarray(mu)
        sigma_1 = np.asarray(sigma_1)
        sigma_2 = np.asarray(sigma_2)

        sigma = np.sqrt(sigma_1**2 + sigma_2**2)
        return_value: npt.NDArray[np.float_] = (
            x
            / (sigma**2)
            * np.exp(-(mu**2 + x**2) / (2 * sigma**2))
            * np.i0(x * mu / (sigma**2))
        )
        return return_value


class PairwiseDistance2dIdenticalSigma(stats.rv_continuous):
    """
    A random variable describing the distribution
    of pair distances for two normal distributed point clouds in 2D.

    The continuous distribution class inherits from scipy.stats.rv_continuous
    a set of methods and is defined by overriding the _pdf method.

    For theoretical background see [1]_.

    Parameters
    ----------
    x : npt.ArrayLike
        distance
    mu : npt.ArrayLike
        Distance between the point cloud center positions.
    sigma : npt.ArrayLike
        Standard deviation for both point clouds.

    References
    ----------
    .. [1] L. Stirling Churchman, Henrik Flyvbjerg, James A. Spudich,
       A Non-Gaussian Distribution Quantifies Distances Measured with
       Fluorescence Localization Techniques.
       Biophysical Journal 90 (2), 2006, 668-671,
       doi.org/10.1529/biophysj.105.065599.
    """

    def _pdf(
        self, x: npt.ArrayLike, mu: npt.ArrayLike, sigma: npt.ArrayLike
    ) -> npt.NDArray[np.float_]:
        x = np.asarray(x)
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)

        return_value: npt.NDArray[np.float_] = (
            x
            / (sigma**2)
            * np.exp(-(mu**2 + x**2) / (2 * sigma**2))
            * np.i0(x * mu / (sigma**2))
        )
        return return_value


class PairwiseDistance3d(stats.rv_continuous):
    """
    A random variable describing the distribution
    of pair distances for two normal distributed point clouds in 3D.

    The continuous distribution class inherits from scipy.stats.rv_continuous
    a set of methods and is defined by overriding the _pdf method.

    For theoretical background see [1]_.

    Parameters
    ----------
    x : npt.ArrayLike
        distance
    mu : npt.ArrayLike
        Distance between the point cloud center positions.
    sigma_1 : npt.ArrayLike
        Standard deviation for one point cloud.
    sigma_2 : npt.ArrayLike
        Standard deviation for the other point cloud.

    References
    ----------
    .. [1] L. Stirling Churchman, Henrik Flyvbjerg, James A. Spudich,
       A Non-Gaussian Distribution Quantifies Distances Measured with
       Fluorescence Localization Techniques.
       Biophysical Journal 90 (2), 2006, 668-671,
       doi.org/10.1529/biophysj.105.065599.
    """

    def _pdf(
        self,
        x: npt.ArrayLike,
        mu: npt.ArrayLike,
        sigma_1: npt.ArrayLike,
        sigma_2: npt.ArrayLike,
    ) -> npt.NDArray[np.float_]:
        x = np.asarray(x)
        mu = np.asarray(mu)
        sigma_1 = np.asarray(sigma_1)
        sigma_2 = np.asarray(sigma_2)

        sigma = np.sqrt(sigma_1**2 + sigma_2**2)
        if all(mu == 0):
            return_value: npt.NDArray[np.float_] = (
                np.sqrt(2 / np.pi)
                * x
                / sigma
                * np.exp(-(mu**2 + x**2) / (2 * sigma**2))
                * x
                / (sigma**2)
            )
        else:
            return_value = (
                np.sqrt(2 / np.pi)
                * x
                / sigma
                / mu
                * np.exp(-(mu**2 + x**2) / (2 * sigma**2))
                * np.sinh(x * mu / (sigma**2))
            )
        return return_value


class PairwiseDistance1dIdenticalSigmaZeroMu(stats.rv_continuous):
    """
    A random variable describing the distribution
    of pair distances for two normal distributed point clouds in 1D.

    The continuous distribution class inherits from scipy.stats.rv_continuous
    a set of methods and is defined by overriding the _pdf method.

    For theoretical background see [1]_.

    Parameters
    ----------
    x : npt.ArrayLike
        distance
    sigma : npt.ArrayLike
        Standard deviation for point clouds

    References
    ----------
    .. [1] L. Stirling Churchman, Henrik Flyvbjerg, James A. Spudich,
       A Non-Gaussian Distribution Quantifies Distances Measured with
       Fluorescence Localization Techniques.
       Biophysical Journal 90 (2), 2006, 668-671,
       doi.org/10.1529/biophysj.105.065599.
    """

    def _pdf(self, x: npt.ArrayLike, sigma: npt.ArrayLike) -> npt.NDArray[np.float_]:
        x = np.asarray(x)
        sigma = np.asarray(sigma)

        return_value: npt.NDArray[np.float_] = (
            np.sqrt(2 / np.pi) / sigma * np.exp(-(x**2) / (2 * sigma**2))
        )
        return return_value


class PairwiseDistance2dIdenticalSigmaZeroMu(stats.rv_continuous):
    """
    A random variable describing the distribution
    of pair distances for two normal distributed point clouds in 2D.

    The continuous distribution class inherits from scipy.stats.rv_continuous
    a set of methods and is defined by overriding the _pdf method.

    For theoretical background see [1]_.

    Parameters
    ----------
    x : npt.ArrayLike
        distance
    sigma : npt.ArrayLike
        Standard deviation for point clouds

    References
    ----------
    .. [1] L. Stirling Churchman, Henrik Flyvbjerg, James A. Spudich,
       A Non-Gaussian Distribution Quantifies Distances Measured with
       Fluorescence Localization Techniques.
       Biophysical Journal 90 (2), 2006, 668-671,
       doi.org/10.1529/biophysj.105.065599.
    """

    def _pdf(self, x: npt.ArrayLike, sigma: npt.ArrayLike) -> npt.NDArray[np.float_]:
        x = np.asarray(x)
        sigma = np.asarray(sigma)

        return_value: npt.NDArray[np.float_] = (
            x / (sigma**2) * np.exp(-(x**2) / (2 * sigma**2))
        )
        return return_value


class PairwiseDistance3dIdenticalSigmaZeroMu(stats.rv_continuous):
    """
    A random variable describing the distribution
    of pair distances for two normal distributed point clouds in 3D.

    The continuous distribution class inherits from scipy.stats.rv_continuous
    a set of methods and is defined by overriding the _pdf method.

    For theoretical background see [1]_.

    Parameters
    ----------
    x : npt.ArrayLike
        distance
    sigma : npt.ArrayLike
        Standard deviation for point clouds

    References
    ----------
    .. [1] L. Stirling Churchman, Henrik Flyvbjerg, James A. Spudich,
       A Non-Gaussian Distribution Quantifies Distances Measured with
       Fluorescence Localization Techniques.
       Biophysical Journal 90 (2), 2006, 668-671,
       doi.org/10.1529/biophysj.105.065599.
    """

    def _pdf(self, x: npt.ArrayLike, sigma: npt.ArrayLike) -> npt.NDArray[np.float_]:
        x = np.asarray(x)
        sigma = np.asarray(sigma)

        return_value: npt.NDArray[np.float_] = (
            np.sqrt(2 / np.pi)
            * x
            / sigma
            * np.exp(-(x**2) / (2 * sigma**2))
            * x
            / (sigma**2)
        )
        return return_value


class _DistributionFits:
    """
    Handle for distribution fits.

    This class is typically instantiated by LocalizationPrecision methods.
    It holds the statistical parameters derived by fitting the result
    distributions using MLE (scipy.stats).

    Parameters
    ----------
    analyis_class : LocalizationPrecision
        The analysis class with result data to fit.

    Attributes
    ----------
    analysis_class : LocalizationPrecision
        The analysis class with result data to fit.
    pairwise_distribution : Pairwise_distance_distribution_2d
        Continuous distribution function used to fit Position_distances
    parameters : list[str]
        Distribution parameters.

    Note
    ----
    Attributes for fit parameter are generated dynamically, named as
    loc_property + distribution parameters and listed in parameters.
    """

    def __init__(self, analysis_class: LocalizationPrecision) -> None:
        self.analysis_class: LocalizationPrecision = analysis_class
        self.distribution: stats.rv_continuous | None = None
        self._dist_parameters: list[str] | None = None
        self.parameters: list[str] = []

        # continuous distributions
        if self.analysis_class.results is None:
            self.pairwise_distribution = None
        else:
            delta_columns = [
                c for c in self.analysis_class.results.columns if "position_delta" in c
            ]
            if len(delta_columns) == 1:
                self.pairwise_distribution = PairwiseDistance1dIdenticalSigmaZeroMu(
                    name="pairwise", a=0.0
                )
            elif len(delta_columns) == 2:
                self.pairwise_distribution = PairwiseDistance2dIdenticalSigmaZeroMu(
                    name="pairwise", a=0.0
                )
            elif len(delta_columns) == 3:
                self.pairwise_distribution = PairwiseDistance3dIdenticalSigmaZeroMu(
                    name="pairwise", a=0.0
                )
            # a is the lower bound of the support of the distribution
            # self.pairwise_distribution = PairwiseDistance2dIdenticalSigma(name='pairwise') also works but is very slow.

    def fit(self, loc_property: str = "position_distance", **kwargs: Any) -> None:
        """
        Fit distributions of results using a MLE fit (scipy.stats) and provide
        fit results.

        Parameters
        ----------
        loc_property
            The property for which to fit an appropriate distribution
        kwargs
            Other parameters passed to the `distribution.fit()` method.
        """
        if self.analysis_class.results is None:
            raise ValueError("Compute results before fitting.")

        # prepare parameters
        if "position_delta_" in loc_property:
            self.distribution = stats.norm
            self._dist_parameters = [
                (loc_property + "_" + param) for param in ["loc", "scale"]
            ]
        elif loc_property == "position_distance":
            self.distribution = self.pairwise_distribution
            self._dist_parameters = [
                (loc_property + "_" + param) for param in ["sigma", "loc", "scale"]
            ]
        else:
            raise TypeError("Unknown localization property.")

        for param in self._dist_parameters:
            if param not in self.parameters:
                self.parameters.append(param)

        # MLE fit of distribution on data
        if "position_delta_" in loc_property:
            fit_results = self.distribution.fit(
                self.analysis_class.results[loc_property].values, **kwargs
            )
        elif loc_property == "position_distance":
            fit_results = self.distribution.fit(
                self.analysis_class.results[loc_property].values,
                **dict(dict(floc=0, fscale=1), **kwargs),
            )
        else:
            raise TypeError("Unknown localization property.")

        for parameter, result in zip(self._dist_parameters, fit_results):
            setattr(self, parameter, result)

    def plot(
        self,
        ax: mpl.axes.Axes | None = None,
        loc_property: str = "position_distance",
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide plot as :class:`matplotlib.axes.Axes` object showing the
        probability distribution functions of fitted results.

        Parameters
        ----------
        ax
            The axes on which to show the image.
        loc_property
            The property for which to plot the distribution fit.
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.distribution is None:
            return ax

        # plot fit curve
        if "position_delta_" in loc_property:
            _loc = getattr(self, loc_property + "_loc")
            _scale = getattr(self, loc_property + "_scale")

            x_values = np.linspace(
                stats.norm.ppf(0.01, loc=_loc, scale=_scale),
                stats.norm.ppf(0.99, loc=_loc, scale=_scale),
                100,
            )
            ax.plot(
                x_values,
                stats.norm.pdf(x_values, loc=_loc, scale=_scale),
                "r-",
                lw=3,
                alpha=0.6,
                label="fitted pdf",
                **kwargs,
            )

        elif loc_property == "position_distance":
            if isinstance(
                self.pairwise_distribution, PairwiseDistance2dIdenticalSigma
            ):  # pragma: no cover
                _sigma = self.position_distance_sigma  # type: ignore
                _mu = self.position_distance_mu  # type: ignore
                x_values = np.linspace(
                    self.pairwise_distribution.ppf(0.01, mu=_mu, sigma=_sigma),
                    self.pairwise_distribution.ppf(0.99, mu=_mu, sigma=_sigma),
                    100,
                )
                ax.plot(
                    x_values,
                    self.pairwise_distribution.pdf(x_values, mu=_mu, sigma=_sigma),
                    "r-",
                    lw=3,
                    alpha=0.6,
                    label="fitted pdf",
                    **kwargs,
                )
            elif isinstance(
                self.pairwise_distribution,
                (
                    PairwiseDistance1dIdenticalSigmaZeroMu,
                    PairwiseDistance2dIdenticalSigmaZeroMu,
                    PairwiseDistance3dIdenticalSigmaZeroMu,
                ),
            ):
                _sigma = self.position_distance_sigma  # type: ignore
                x_values = np.linspace(
                    self.pairwise_distribution.ppf(0.01, sigma=_sigma),
                    self.pairwise_distribution.ppf(0.99, sigma=_sigma),
                    100,
                )
                ax.plot(
                    x_values,
                    self.pairwise_distribution.pdf(x_values, sigma=_sigma),
                    "r-",
                    lw=3,
                    alpha=0.6,
                    label="fitted pdf",
                    **kwargs,
                )
            else:
                raise NotImplementedError(
                    "pairwise_distribution function has not been implemented for plotting "
                    "position distances."
                )

        return ax

    def parameter_dict(self) -> dict[str, float]:
        """Dictionary of fitted parameters."""
        return {k: self.__dict__[k] for k in self.parameters}
