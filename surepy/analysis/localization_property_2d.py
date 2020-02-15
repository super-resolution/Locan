"""
Analyze the distribution of a localization property in 2D.

Analyze the distribution of a localization property as function of two other localization properties in 2D.
E.g. looking at how the local background is distributed over localization coordinates helps to characterize the
illumination profile in SMLM experiments.
"""
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

from surepy.render.render2d import render_2d_mpl
from surepy.analysis.analysis_base import _Analysis, _list_parameters
from surepy.render.render2d import histogram


__all__ = ['LocalizationProperty2d']


##### The algorithms

# model fit function in 2d
def _gauss_2d(x, y, amplitude, center_x, center_y, sigma_x, sigma_y):
    """
    2D Gauss function of variables (x, y).

    Parameters
    ----------
    x : float or array-like
        all x values
    y : float or array-like
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
    float or ndarray
    """
    x_ = np.asarray(x)
    y_ = np.asarray(y)
    amplitude, center_x, center_y, sigma_x, sigma_y = (np.float(number) for number in
                                                       (amplitude, center_x, center_y, sigma_x, sigma_y))
    return amplitude * np.exp(-((x_-center_x)**2/(2.*sigma_x**2) + (y_-center_y)**2/(2.*sigma_y**2)))


def _fit_image_copy(image, bin_edges):
    """
    Fit 2D Gauss function to image data.

    Parameters
    ----------
    image : ndarray
        binned localization data
    bin_edges : ndarray
        bin_edges as returned from histogram().

    Returns
    -------
    lmfit.model.ModelResult object
        The fit results.
    """
    # prepare 1D lmfit model from 2D model function
    def model_function(points, amplitude=1, center_x=0, center_y=0, sigma_x=1, sigma_y=1):
        return np.ravel(_gauss_2d(*points.T, amplitude, center_x, center_y, sigma_x, sigma_y))

    model = Model(model_function, nan_policy='omit')
    # print(model.param_names, model.independent_vars)

    # simple definition of range (which is not strictly the same as outter `bin_edges`).
    range_ = np.array([bin_edges[0][[0, -1]], bin_edges[1][[0, -1]]])

    # prepare data
    xx, yy = np.meshgrid(bin_edges[0][1:], bin_edges[1][1:])
    data = np.empty((np.product(image.shape), 3))
    data[:, 0] = xx.flatten()
    data[:, 1] = yy.flatten()
    data[:, 2] = image.flatten()

    data[:, 2][data[:, 2] == 0] = np.nan

    # instantiate lmfit Parameters
    params = Parameters()
    params.add('amplitude', value=np.amax(image))
    centers = np.add(range_[:, 0], np.diff(range_).flatten() / 2)
    params.add('center_x', value=centers[0])
    params.add('center_y', value=centers[1])
    sigmas = np.diff(range_).ravel() / 4
    params.add('sigma_x', value=sigmas[0])
    params.add('sigma_y', value=sigmas[1])

    # fit
    model_result = model.fit(data[:, 2], points=data[:, 0:2], params=params)

    mask = np.isfinite(data[:, 2])
    best_fit_with_nan = np.copy(data[:, 2])
    best_fit_with_nan[mask] = model_result.best_fit

    return model_result, best_fit_with_nan


def _fit_image(data, range):
    """
    Fit 2D Gauss function to image data.

    Parameters
    ----------
    data : ndarray of shape (3, n_image_values)
        arrays with corresponding values for x, y, z
    range : ndarray
        range as returned from histogram().

    Returns
    -------
    lmfit.model.ModelResult object
        The fit results.
    """
    # prepare 1D lmfit model from 2D model function
    def model_function(points, amplitude=1, center_x=0, center_y=0, sigma_x=1, sigma_y=1):
        return np.ravel(_gauss_2d(*points.T, amplitude, center_x, center_y, sigma_x, sigma_y))

    model = Model(model_function, nan_policy='omit')
    # print(model.param_names, model.independent_vars)

    # instantiate lmfit Parameters
    params = Parameters()
    params.add('amplitude', value=np.amax(data[2]))
    centers = np.add(range[:, 0], np.diff(range).flatten() / 2)
    params.add('center_x', value=centers[0])
    params.add('center_y', value=centers[1])
    sigmas = np.diff(range).ravel() / 4
    params.add('sigma_x', value=sigmas[0])
    params.add('sigma_y', value=sigmas[1])

    # fit
    model_result = model.fit(data[2], points=data[0:2].T, params=params)

    return model_result


def _localization_property2d(locdata, loc_properties=None, other_property=None,
                             bins=None, bin_size=10, range=None, rescale=None):
    # bin localization data
    img, range, bin_edges, label = histogram(locdata, loc_properties, other_property, bins, bin_size, range, rescale)

    # prepare one-dimensional data
    xx, yy = np.meshgrid(bin_edges[0][1:], bin_edges[1][1:])

    # eliminate image zeros
    positions = np.nonzero(img)

    data_0 = xx[positions].flatten()
    data_1 = yy[positions].flatten()
    data_2 = img[positions].flatten()

    data = np.stack((data_0, data_1, data_2))

    model_result = _fit_image(data, range)

    Results = namedtuple('results', 'img range bin_edges label model_result')
    results = Results(img, range, bin_edges, label, model_result)
    return results


##### The specific analysis classes

class LocalizationProperty2d(_Analysis):
    """
    Analyze 2d distribution of histogram for two localization properties.

    Fit a two dimensional Gauss distribution.

    Parameters
    ----------
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    loc_properties : list or None
        Localization properties to be grouped into bins. If None The coordinate_values of locdata are used.
    other_property : str or None
        Localization property (columns in locdata.data) that is averaged in each pixel. If None localization counts are
        shown.
    bins : sequence or int or None
        The bin specification as defined in numpy.histogramdd:
            A sequence of arrays describing the monotonically increasing bin edges along each dimension.
            The number of bins for each dimension (nx, ny, … =bins)
            The number of bins for all dimensions (nx=ny=…=bins).
    bin_size : float or tuple, list, ndarray with with length equal to that of range (or the number of loc_properties).
        Number of bins to be used in all or each dimension for which a range is provided.
        The bin size in units of locdata coordinate units. Either bins or bin_size must be specified but not both.
    range : tuple with shape (dimension, 2) or None or 'zero'
        ((min_x, max_x), (min_y, max_y), ...) range for each coordinate;
        for None (min, max) range are determined from data;
        for 'zero' (0, max) range with max determined from data.
    rescale : True, tuple, False or None, 'equal', or 'unity.
        Rescale intensity values to be within percentile of max and min intensities
        (tuple with upper and lower bounds provided in percent).
        For True intensity values are rescaled to the min and max possible values of the given representation.
        For 'equal' intensity values are rescaled by histogram equalization.
        For 'unity' intensity values are rescaled to (0, 1).
        For None or False no rescaling occurs.

    Attributes
    ----------
    count : int
        A counter for counting instantiations (class attribute).
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : Metadata protobuf message
        Metadata about the current analysis routine.
    results : lmfit.model.ModelResult object
        Computed fit results.
    """
    def __init__(self, meta=None,
                 loc_properties=None, other_property='local_background',
                 bins=None, bin_size=100, range=None, rescale=None,
                 ):
        super().__init__(meta=meta, loc_properties=loc_properties, other_property=other_property,
                         bins=bins, bin_size=bin_size, range=range, rescale=rescale)
        self.results = None

    def compute(self, locdata=None):
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
        self.results = _localization_property2d(locdata=locdata, **self.parameter)
        return self

    def report(self):
        print('Fit results for:\n')
        print(self.results.model_result.fit_report(min_correl=0.25))
        # print(self.results.fit_results.best_values)

        # judge fit parameter
        max_fit_value = self.results.model_result.best_fit.max()
        min_fit_value = self.results.model_result.best_fit.min()
        ratio = (max_fit_value - min_fit_value) / max_fit_value
        print(f'Maximum fit value in image: {max_fit_value:.3f}')
        print(f'Minimum fit value in image: {min_fit_value:.3f}')
        print(f'Fit value variation over image range: {ratio:.2f}')

    def plot(self, ax=None, **kwargs):
        """
        Provide histogram as matplotlib axes object showing plot(results).

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to matplotlib.pyplot.contour().

        Returns
        -------
        matplotlib Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        ax.imshow(self.results.img, cmap='viridis', origin='lower', extent=np.ravel(self.results.range))

        x, y = self.results.bin_edges[0][1:], self.results.bin_edges[1][1:]
        xx, yy = np.meshgrid(x, y)
        zz = np.stack((xx, yy), axis=-1).reshape((np.product(xx.shape), 2))
        z = self.results.model_result.eval(points=zz)

        contourset = ax.contour(x, y, z.reshape((len(y), len(x))), 8, colors='w', **kwargs)
        plt.clabel(contourset, fontsize=9, inline=1)
        ax.set(title=self.parameter['other_property'],
               xlabel=self.results.label[0],
               ylabel=self.results.label[1]
               )

        return ax

    def plot_residuals(self, ax=None, **kwargs):
        """
        Provide histogram as matplotlib axes object showing plot(results).

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to matplotlib.pyplot.contour().

        Returns
        -------
        matplotlib Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        x, y = self.results.bin_edges[0][1:], self.results.bin_edges[1][1:]
        xx, yy = np.meshgrid(x, y)
        zz = np.stack((xx, yy), axis=-1).reshape((np.product(xx.shape), 2))
        z = self.results.model_result.eval(points=zz)

        residuals = np.where(self.results.img == 0, np.nan, z.reshape((len(y), len(x))) - self.results.img)
        ax.imshow(residuals, cmap='coolwarm', origin='lower', extent=np.ravel(self.results.range))

        # contourset = ax.contour(x, y, z.reshape((len(y), len(x))), 8, colors='w', **kwargs)
        # plt.clabel(contourset, fontsize=9, inline=1)
        ax.set(title=self.parameter['other_property'],
               xlabel=self.results.label[0],
               ylabel=self.results.label[1]
               )

        return ax

    def plot_deviation_from_mean(self, ax=None, **kwargs):
        """
        Provide histogram as matplotlib axes object showing plot(results).

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to matplotlib.pyplot.contour().

        Returns
        -------
        matplotlib Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        positions = np.nonzero(self.results.img)
        mean_value = self.results.img[positions].mean()
        deviations = np.where(self.results.img == 0, np.nan, self.results.img - mean_value)
        ax.imshow(deviations, cmap='coolwarm', origin='lower', extent=np.ravel(self.results.range))

        ax.set(title=f"{self.parameter['other_property']} - deviation from mean",
               xlabel=self.results.label[0],
               ylabel=self.results.label[1]
               )

        return ax

    def plot_deviation_from_median(self, ax=None, **kwargs):
        """
        Provide histogram as matplotlib axes object showing plot(results).

        Parameters
        ----------
        ax : matplotlib axes
            The axes on which to show the image

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters passed to matplotlib.pyplot.contour().

        Returns
        -------
        matplotlib Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        positions = np.nonzero(self.results.img)
        median_value = np.median(self.results.img[positions])
        deviations = np.where(self.results.img == 0, np.nan, self.results.img - median_value)
        ax.imshow(deviations, cmap='coolwarm', origin='lower', extent=np.ravel(self.results.range))

        ax.set(title=f"{self.parameter['other_property']} - deviation from median",
               xlabel=self.results.label[0],
               ylabel=self.results.label[1]
               )

        return ax