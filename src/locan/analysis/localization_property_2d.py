"""
Analyze the distribution of a localization property in 2D.

Analyze the distribution of a localization property as function of two other
localization properties in 2D.
E.g. looking at how the local background is distributed over localization
coordinates helps to characterize the
illumination profile in SMLM experiments.
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    import boost_histogram as bh
    import matplotlib as mpl

    from locan.data.locdata import LocData

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from lmfit import Model, Parameters
from lmfit.model import ModelResult

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis
from locan.configuration import COLORMAP_DEFAULTS
from locan.data.aggregate import Bins, histogram
from locan.visualize.transform import Trafo, adjust_contrast

__all__: list[str] = ["LocalizationProperty2d"]

logger = logging.getLogger(__name__)


# The algorithms


class FitImageResults(NamedTuple):
    image: npt.NDArray[Any]
    bins: Bins
    label: list[str]
    model_result: ModelResult


# model fit function in 2d
def _gauss_2d(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    amplitude: float,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
) -> float | npt.NDArray[np.float_]:
    """
    2D Gauss function of variables (x, y).

    Parameters
    ----------
    x
        all x values
    y
        all y values
    amplitude
        amplitude
    center_x
        x-shift
    center_y
        y-shift
    sigma_x
        standard deviation x
    sigma_y
        standard deviation y

    Returns
    -------
    float | npt.NDArray[np.float_]
    """
    x_ = np.asarray(x)
    y_ = np.asarray(y)
    amplitude, center_x, center_y, sigma_x, sigma_y = (  # type: ignore[assignment]
        np.float64(number)
        for number in (amplitude, center_x, center_y, sigma_x, sigma_y)
    )
    return_value: float | npt.NDArray[np.float_] = amplitude * np.exp(
        -(
            (x_ - center_x) ** 2 / (2.0 * sigma_x**2)
            + (y_ - center_y) ** 2 / (2.0 * sigma_y**2)
        )
    )
    return return_value


def _fit_image(data: npt.ArrayLike, bin_range: npt.ArrayLike) -> ModelResult:
    """
    Fit 2D Gauss function to image data.

    Parameters
    ----------
    data
        arrays with corresponding values for x, y, z
        of shape (3, n_image_values)
    bin_range
        range as returned from histogram().

    Returns
    -------
    lmfit.model.ModelResult
        The fit results.
    """

    # prepare 1D lmfit model from 2D model function
    def model_function(
        points: npt.ArrayLike,
        amplitude: float = 1,
        center_x: float = 0,
        center_y: float = 0,
        sigma_x: float = 1,
        sigma_y: float = 1,
    ) -> npt.NDArray[Any]:
        points = np.asarray(points)
        return np.ravel(
            _gauss_2d(*points.T, amplitude, center_x, center_y, sigma_x, sigma_y)  # type: ignore[call-arg]
        )

    model = Model(model_function, nan_policy="omit")
    # print(model.param_names, model.independent_vars)

    data = np.asarray(data)
    bin_range = np.asarray(bin_range)

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
    locdata: LocData,
    loc_properties: str | Sequence[str] | None = None,
    other_property: str | None = None,
    bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
    n_bins: int | Sequence[int] | None = None,
    bin_size: float | Sequence[float] | Sequence[Sequence[float]] | None = 10,
    bin_edges: Sequence[float] | Sequence[Sequence[float]] | None = None,
    bin_range: tuple[float, float]
    | Sequence[float]
    | Sequence[Sequence[float]]
    | Literal["zero", "link"]
    | None = None,
    rescale: int | str | Trafo | Callable[..., Any] | bool | None = None,
) -> FitImageResults:
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
    results = FitImageResults(img, bins, label, model_result)
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
    loc_properties : Sequence[str] | None
        Localization properties to be grouped into bins.
        If None, the coordinate_values of locdata are used.
    other_property : str | None
        Localization property (columns in locdata.data) that is averaged in each pixel.
        If None, the localization counts are shown.
    bins : Bins | boost_histogram.axis.Axis | boost_histogram.axis.AxesTuple | None
        The bin specification as defined in :class:`Bins`
    bin_edges : Sequence[float] | Sequence[Sequence[float]] | None
        Bin edges for all or each dimension
        with shape (dimension, n_bin_edges).
    bin_range : tuple[float, float] | Sequence[float] | Sequence[Sequence[float]] | str | None
        Minimum and maximum edge for all or each dimensions
        with shape (2,) or (dimension, 2).
        If None (min, max) ranges are determined from data and returned;
        if 'zero' (0, max) ranges with max determined from data are returned.
        if 'link' (min_all, max_all) ranges with min and max determined from
        all combined data are returned.
    n_bins : int | Sequence[int] | None
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size : float | Sequence[float] | Sequence[Sequence[float]] | None
        The size of bins for all or each bin and for all or each dimension
        with shape (dimension,) or (dimension, n_bins).
        5 would describe bin_size of 5 for all bins in all dimensions.
        ((2, 5),) yield bins of size (2, 5) for one dimension.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other
        dimension.
        ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and
        (1, 3) for the other dimension.
        To specify arbitrary sequence of `bin_size` use `bin_edges` instead.
    rescale : int | str | locan.Trafo | Callable[..., Any] | bool | None
        Transformation as defined in :class:`locan.Trafo` or by
        transformation function.
        For None or False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min
        intensities.
        For 'equal' intensity values are rescaled by histogram equalization.

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : lmfit.model.ModelResult
        Computed fit results.
    """

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        loc_properties: Sequence[str] | None = None,
        other_property: str | None = "local_background",
        bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
        n_bins: int | Sequence[int] | None = None,
        bin_size: float | Sequence[float] | Sequence[Sequence[float]] | None = 100,
        bin_edges: Sequence[float] | Sequence[Sequence[float]] | None = None,
        bin_range: tuple[float, float]
        | Sequence[float]
        | Sequence[Sequence[float]]
        | str
        | None = None,
        rescale: int | str | Trafo | Callable[..., Any] | bool | None = None,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.results = None

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

        self.results = _localization_property2d(locdata=locdata, **self.parameter)
        return self

    def report(self) -> None:
        if self.results is None:
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

    def plot(self, ax: mpl.axes.Axes | None = None, **kwargs: Any) -> mpl.axes.Axes:
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing plot(results).

        Parameters
        ----------
        ax
            The axes on which to show the image
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.contour`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
            return ax

        ax.imshow(
            self.results.image.T,
            cmap="viridis",
            origin="lower",
            extent=np.ravel(self.results.bins.bin_range),  # type: ignore
        )

        x, y = self.results.bins.bin_centers
        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = np.stack((xx, yy), axis=-1).reshape((np.prod(xx.shape), 2))
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

    def plot_residuals(
        self, ax: mpl.axes.Axes | None = None, **kwargs: Any
    ) -> mpl.axes.Axes:
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing plot(results).

        Parameters
        ----------
        ax
            The axes on which to show the image
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.contour`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
            return ax

        x, y = self.results.bins.bin_centers
        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = np.stack((xx, yy), axis=-1).reshape((np.prod(xx.shape), 2))
        z = self.results.model_result.eval(points=zz)

        residuals = np.where(
            self.results.image == 0,
            np.nan,
            z.reshape((len(x), len(y))) - self.results.image,
        )
        max_absolute_value = max([abs(np.nanmin(residuals)), abs(np.nanmax(residuals))])
        ax.imshow(
            residuals.T,
            cmap=COLORMAP_DEFAULTS["DIVERGING"],
            origin="lower",
            extent=np.ravel(self.results.bins.bin_range),  # type: ignore
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

    def plot_deviation_from_mean(
        self, ax: mpl.axes.Axes | None = None
    ) -> mpl.axes.Axes:
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing
        plot(results).

        Parameters
        ----------
        ax
            The axes on which to show the image

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
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
            cmap=COLORMAP_DEFAULTS["DIVERGING"],
            origin="lower",
            extent=np.ravel(self.results.bins.bin_range),  # type: ignore
            vmin=(-max_absolute_value),
            vmax=max_absolute_value,
        )

        ax.set(
            title=f"{self.parameter['other_property']} - deviation from mean",
            xlabel=self.results.label[0],
            ylabel=self.results.label[1],
        )

        return ax

    def plot_deviation_from_median(
        self, ax: mpl.axes.Axes | None = None
    ) -> mpl.axes.Axes:
        """
        Provide histogram as :class:`matplotlib.axes.Axes` object showing
        plot(results).

        Parameters
        ----------
        ax
            The axes on which to show the image

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
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
            cmap=COLORMAP_DEFAULTS["DIVERGING"],
            origin="lower",
            extent=np.ravel(self.results.bins.bin_range),  # type: ignore
            vmin=(-max_absolute_value),
            vmax=max_absolute_value,
        )

        ax.set(
            title=f"{self.parameter['other_property']} - deviation from median",
            xlabel=self.results.label[0],
            ylabel=self.results.label[1],
        )

        return ax
