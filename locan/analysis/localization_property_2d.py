"""
Analyze the distribution of a localization property in 2D.

Analyze the distribution of a localization property as function of two other localization properties in 2D.
E.g. looking at how the local background is distributed over localization coordinates helps to characterize the
illumination profile in SMLM experiments.
"""
from __future__ import annotations

import logging
import sys
from collections import namedtuple

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt  # noqa: F401
from lmfit import Model, Parameters

from locan.analysis.analysis_base import _Analysis
from locan.configuration import COLORMAP_DIVERGING
from locan.data.aggregate import histogram
from locan.visualize.transform import adjust_contrast

__all__: list[str] = ["LocalizationProperty2d"]

logger = logging.getLogger(__name__)


# The algorithms


# model fit function in 2d
def _gauss_2d(x, y, amplitude, center_x, center_y, sigma_x, sigma_y):
    """
    2D Gauss function of variables (x, y).

    Parameters
    ----------
    x : float, npt.ArrayLike
        all x values
    y : float, npt.ArrayLike
        all y values
    amplitude : float
        amplitude
    center_x : float
        x-shift
    center_y : float
        y-shift
    sigma_x : float
        standard deviation x
    sigma_y : float
        standard deviation y

    Returns
    -------
    float, numpy.ndarray
    """
    x_ = np.asarray(x)
    y_ = np.asarray(y)
    amplitude, center_x, center_y, sigma_x, sigma_y = (
        np.float64(number)
        for number in (amplitude, center_x, center_y, sigma_x, sigma_y)
    )
    return amplitude * np.exp(
        -(
            (x_ - center_x) ** 2 / (2.0 * sigma_x**2)
            + (y_ - center_y) ** 2 / (2.0 * sigma_y**2)
        )
    )


def _fit_image(data, bin_range):
    """
    Fit 2D Gauss function to image data.

    Parameters
    ----------
    data : npt.ArrayLike
        arrays with corresponding values for x, y, z
        of shape (3, n_image_values)
    bin_range : npt.ArrayLike
        range as returned from histogram().

    Returns
    -------
    lmfit.model.ModelResult object
        The fit results.
    """

    # prepare 1D lmfit model from 2D model function
    def model_function(
        points, amplitude=1, center_x=0, center_y=0, sigma_x=1, sigma_y=1
    ):
        return np.ravel(
            _gauss_2d(*points.T, amplitude, center_x, center_y, sigma_x, sigma_y)
        )

    model = Model(model_function, nan_policy="omit")
    # print(model.param_names, model.independent_vars)

    # instantiate lmfit Parameters
    params = Parameters()
    params.add("amplitude", value=np.amax(data[2]))
    centers = np.add(bin_range[:, 0], np.diff(bin_range).flatten() / 2)
    params.add("center_x", value=centers[0])
    params.add("center_y", value=centers[1])
    sigmas = np.diff(bin_range).ravel() / 4
    params.add("sigma_x", value=sigmas[0])
    params.add("sigma_y", value=sigmas[1])

    # fit
    model_result = model.fit(data[2], points=data[0:2].T, params=params)

    return model_result


def _localization_property2d(
    locdata,
    loc_properties=None,
    other_property=None,
    bins=None,
    n_bins=None,
    bin_size=10,
    bin_edges=None,
    bin_range=None,
    rescale=None,
):
    # bin localization data
    img, bins, label = histogram(
        locdata=locdata,
        loc_properties=loc_properties,
        other_property=other_property,
        bins=bins,
        n_bins=n_bins,
        bin_size=bin_size,
        bin_edges=bin_edges,
        bin_range=bin_range,
    )
    img = adjust_contrast(img, rescale)

    # prepare one-dimensional data
    x, y = bins.bin_centers
    xx, yy = np.meshgrid(x, y, indexing="ij")
    # eliminate image zeros
    positions = np.nonzero(~np.isnan(img))

    data_0 = xx[positions].flatten()
    data_1 = yy[positions].flatten()
    data_2 = img[positions].flatten()

    data = np.stack((data_0, data_1, data_2))

    model_result = _fit_image(data, np.array(bins.bin_range))

    Results = namedtuple("Results", "image bins label model_result")
    results = Results(img, bins, label, model_result)
    return results


# The specific analysis classes


class LocalizationProperty2d(_Analysis):
    """
    Analyze 2d distribution of histogram for two localization properties.

    Fit a two dimensional Gauss distribution.

    Parameters
    ----------
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    loc_properties : list, None
        Localization properties to be grouped into bins.
        If None, the coordinate_values of locdata are used.
    other_property : str, None
        Localization property (columns in locdata.data) that is averaged in each pixel.
        If None, the localization counts are shown.
    bins : int, sequence, Bins, boost_histogram.axis.Axis, None
        The bin specification as defined in :class:`Bins`
    bin_edges : tuple | list | numpy.ndarray[float] | None
        Array of bin edges with shape (dimension, n_bin_edges)
        for all or each dimension.
    n_bins : int, list, tuple, numpy.ndarray, None
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size : float, list, tuple, numpy.ndarray, None
        The size of bins in units of locdata coordinate units for all or each dimension.
        5 would describe bin_size of 5 for all bins in all dimensions.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other dimension.
        To specify arbitrary sequence of `bin_sizes` use `bin_edges` instead.
    bin_range : tuple, tuple of tuples of float with shape (dimension, 2), None, 'zero'
        The data bin_range to be taken into consideration for all or each dimension.
        ((min_x, max_x), (min_y, max_y), ...) bin_range for each coordinate;
        for None (min, max) bin_range are determined from data;
        for 'zero' (0, max) bin_range with max determined from data.
    rescale : int, str, locan.constants.Trafo, callable, bool, None
        Transformation as defined in Trafo or by transformation function.
        For None or False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min intensities
        For 'equal' intensity values are rescaled by histogram equalization.

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : lmfit.model.ModelResult object
        Computed fit results.
    """

    def __init__(
        self,
        meta=None,
        loc_properties=None,
        other_property="local_background",
        bins=None,
        n_bins=None,
        bin_size=100,
        bin_edges=None,
        bin_range=None,
        rescale=None,
    ):
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.results = None

    def compute(self, locdata=None) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata : LocData
            Localization data.

        Returns
        -------
        Self
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        self.results = _localization_property2d(locdata=locdata, **self.parameter)
        return self

    def report(self):
        if not self:
            logger.warning("No results available")
            return

        print("Fit results for:\n")
        print(self.results.model_result.fit_report(min_correl=0.25))
        # print(self.results.fit_results.best_values)

        # judge fit parameter
        max_fit_value = self.results.model_result.best_fit.max()
        min_fit_value = self.results.model_result.best_fit.min()
        ratio = (max_fit_value - min_fit_value) / max_fit_value
        print(f"Maximum fit value in image: {max_fit_value:.3f}")
        print(f"Minimum fit value in image: {min_fit_value:.3f}")
        print(f"Fit value variation over image range: {ratio:.2f}")

    def plot(self, ax=None, **kwargs):
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing plot(results).

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which to show the image
        kwargs : dict
            Other parameters passed to :func:`matplotlib.pyplot.contour`.

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        ax.imshow(
            self.results.image.T,
            cmap="viridis",
            origin="lower",
            extent=np.ravel(self.results.bins.bin_range),
        )

        x, y = self.results.bins.bin_centers
        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = np.stack((xx, yy), axis=-1).reshape((np.product(xx.shape), 2))
        z = self.results.model_result.eval(points=zz)

        contourset = ax.contour(
            x, y, z.reshape((len(x), len(y))).T, 8, **dict(dict(colors="w"), **kwargs)
        )
        plt.clabel(contourset, fontsize=9, inline=1)
        ax.set(
            title=self.parameter["other_property"],
            xlabel=self.results.label[0],
            ylabel=self.results.label[1],
        )

        return ax

    def plot_residuals(self, ax=None, **kwargs):
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing plot(results).

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which to show the image
        kwargs : dict
            Other parameters passed to :func:`matplotlib.pyplot.contour`.

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        x, y = self.results.bins.bin_centers
        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = np.stack((xx, yy), axis=-1).reshape((np.product(xx.shape), 2))
        z = self.results.model_result.eval(points=zz)

        residuals = np.where(
            self.results.image == 0,
            np.nan,
            z.reshape((len(x), len(y))) - self.results.image,
        )
        max_absolute_value = max([abs(np.nanmin(residuals)), abs(np.nanmax(residuals))])
        ax.imshow(
            residuals.T,
            cmap=COLORMAP_DIVERGING,
            origin="lower",
            extent=np.ravel(self.results.bins.bin_range),
            vmin=(-max_absolute_value),
            vmax=max_absolute_value,
        )

        contourset = ax.contour(
            x, y, z.reshape((len(x), len(y))).T, 8, **dict(dict(colors="w"), **kwargs)
        )
        plt.clabel(contourset, fontsize=9, inline=1)

        ax.set(
            title=self.parameter["other_property"],
            xlabel=self.results.label[0],
            ylabel=self.results.label[1],
        )

        return ax

    def plot_deviation_from_mean(self, ax=None):
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing plot(results).

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which to show the image

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        positions = np.nonzero(self.results.image)
        mean_value = np.nanmean(self.results.image[positions])
        deviations = np.where(
            self.results.image == 0, np.nan, self.results.image - mean_value
        )
        max_absolute_value = max(
            [abs(np.nanmin(deviations)), abs(np.nanmax(deviations))]
        )
        ax.imshow(
            deviations.T,
            cmap=COLORMAP_DIVERGING,
            origin="lower",
            extent=np.ravel(self.results.bins.bin_range),
            vmin=(-max_absolute_value),
            vmax=max_absolute_value,
        )

        ax.set(
            title=f"{self.parameter['other_property']} - deviation from mean",
            xlabel=self.results.label[0],
            ylabel=self.results.label[1],
        )

        return ax

    def plot_deviation_from_median(self, ax=None):
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing plot(results).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to show the image

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if not self:
            return ax

        positions = np.nonzero(self.results.image)
        median_value = np.nanmedian(self.results.image[positions])
        deviations = np.where(
            self.results.image == 0, np.nan, self.results.image - median_value
        )
        max_absolute_value = max(
            [abs(np.nanmin(deviations)), abs(np.nanmax(deviations))]
        )
        ax.imshow(
            deviations.T,
            cmap=COLORMAP_DIVERGING,
            origin="lower",
            extent=np.ravel(self.results.bins.bin_range),
            vmin=(-max_absolute_value),
            vmax=max_absolute_value,
        )

        ax.set(
            title=f"{self.parameter['other_property']} - deviation from median",
            xlabel=self.results.label[0],
            ylabel=self.results.label[1],
        )

        return ax
