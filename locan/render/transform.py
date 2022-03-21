"""
Transform intensity values.

This module provides functions and classes to rescale intensity values.
In addition to the presented functions, rescaling can further be performed by third-party modules like:
1) matplotlib.colors
2) skimage.exposure
3) astropy.visualization

"""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import matplotlib.colors as mcolors
from skimage import exposure


__all__ = ['Trafo', 'HistogramEqualization', 'adjust_contrast']


class Trafo(Enum):
    """
    Standard definitions for intensity transformation.
    """
    NONE = 0
    STANDARDIZE = 1
    """rescale (min, max) to (0, 1)"""
    STANDARDIZE_UINT8 = 2
    """rescale (min, max) to (0, 255)"""
    ZERO = 3
    """(0, max) to (0, 1)"""
    ZERO_UINT8 = 4
    """(0, max) to (0, 255)"""
    EQUALIZE = 5
    """equalize histogram for all values > 0"""
    EQUALIZE_UINT8 = 6
    """equalize histogram for all values > 0 to (0, 255)"""
    EQUALIZE_ALL = 7
    """equalize histogram"""
    EQUALIZE_ALL_UINT8 = 8
    """equalize histogram to (0, 255)"""
    EQUALIZE_0P3 = 9
    """equalize histogram with power = 0.3 for all values > 0"""
    EQUALIZE_0P3_UINT8 = 10
    """equalize histogram with power = 0.3 for all values > 0 to (0, 255)"""
    EQUALIZE_0P3_ALL = 11
    """equalize histogram with power = 0.3"""
    EQUALIZE_0P3_ALL_UINT8 = 12
    """equalize histogram with power = 0.3 to (0, 255)"""


class Transform(ABC):
    """
    Abstract base class for transformation classes.
    """

    @abstractmethod
    def __call__(self, values, clip=True):
        """
        Transform values.

        Parameters
        ----------
        values : array-like
            The input values
        clip : bool
            If `True` values outside the [0:1] range are
            clipped to the [0:1] range.

        Returns
        -------
        numpy.ndarray
            The transformed values.
        """
        raise NotImplementedError

        _values, is_scalar = self.process_value(values)
        self.autoscale_None(_values)  # sets self.vmin, self.vmax if None

        if clip is None:
            clip = self.clip

        # Normalize based on vmin and vmax
        np.subtract(_values, self.vmin, out=_values)
        np.true_divide(_values, self.vmax - self.vmin, out=_values)

        # Clip to the 0 to 1 range
        if clip:
            np.clip(_values, 0., 1., out=_values)

        new_values =_values
        return new_values

    @property
    @abstractmethod
    def inverse(self):
        """A transformation that performs the inverse operation."""
        raise NotImplementedError


class HistogramEqualization(mcolors.Normalize, Transform):
    """
    Histogram equalization with power intensification as described in [1]_.

    The transformation function is f(a, p) according to:
    .. math::
       \\frac{f(a) - f(a_min)} {f(a_max) - f(a_min)} =
       \\frac{\\int_{a_min}^{a}{h^p(a') da'}} {\\int_{a_min}^{a_max}{h^p(a') da'}}

    Here, :math:`a` is an intensity value, :math:`p` the power (a parameter), and :math:`h(a)` the histogram of
    intensities.

    References
    ----------
    .. [1] Yaroslavsky, L. (1985) Digital picture processing. Springer, Berlin.

    Parameters
    ----------
    reference : array-like
        The data values to define the transformation function. If None values in call are used.
    power : float
        The ``power`` parameter used in the above formula.
    n_bins : int
        Number of bins used to compute the intensity histogram.
    mask : array-like[bool]
        A bool mask with shape equal to that of values. If reference is None it is set to values[mask].
        The transformation determined from reference is then applied to the all values.
    """

    def __init__(self, vmin=None, vmax=None, reference=None, power=1, n_bins=256, mask=None):
        super().__init__(vmin=vmin, vmax=vmax)
        self.reference = reference
        self.power = power
        self.n_bins = n_bins
        self.mask = mask

    def __call__(self, values):
        """
        Histogram equalization with power intensification.

        Parameters
        ----------
        values : array-like
            The input values.
        """
        _values, is_scalar = self.process_value(values)
        self.autoscale_None(_values)  # sets self.vmin, self.vmax if None

        # Normalize based on vmin and vmax
        np.subtract(_values, self.vmin, out=_values)
        np.true_divide(_values, self.vmax - self.vmin, out=_values)

        # Clip to the 0 to 1 range
        np.clip(_values, 0., 1., out=_values)

        if self.reference is None:
            _reference = _values[self.mask]
        else:
            _reference = self.reference

        data, bin_edges = np.histogram(_reference, bins=self.n_bins, range=(0, 1))
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        cdf = np.cumsum(data**self.power)
        cdf = cdf / cdf[-1]
        new_values = np.interp(_values, bin_centers, cdf)
        return new_values

    @property
    def inverse(self, values):
        """A Transformation object that performs the inverse operation."""
        raise NotImplementedError


def adjust_contrast(image, rescale=True, **kwargs):
    """
    Adjust contrast of image by a predefined transformation:

    Parameters
    ----------
    image : array-like
        Values to be adjusted
    rescale : int, str, locan.constants.Trafo, callable, bool, None
        Transformation as defined in Trafo or by transformation function.
        For None or False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min intensities
        For 'equal' intensity values are rescaled by histogram equalization.
    kwargs : dict
        Other parameters that are passed to the specific Transformation class.

    Returns
    -------
    numpy.ndarray
    """
    if rescale is None or rescale is False or rescale is Trafo.NONE or rescale == 0:
        new_image = image

    elif rescale is True or rescale == 1 or rescale is Trafo.STANDARDIZE \
            or (isinstance(rescale, str) and rescale.upper() == Trafo.STANDARDIZE.name):
        new_image = exposure.rescale_intensity(image * 1., **kwargs)

    elif rescale == 2 or rescale is Trafo.STANDARDIZE_UINT8 \
            or (isinstance(rescale, str) and rescale.upper() == Trafo.STANDARDIZE_UINT8.name):
        new_image = exposure.rescale_intensity(image,
                                               **dict(dict(out_range=(0, 255)), **kwargs)
                                               ).astype(np.uint8)

    elif rescale == 3 or rescale is Trafo.ZERO \
            or (isinstance(rescale, str) and rescale.upper() == Trafo.ZERO.name):
        new_image = exposure.rescale_intensity(image * 1.,
                                               **dict(dict(in_range=(0, np.nanmax(image)),
                                                           out_range=(0, 1)), **kwargs)
                                               )

    elif rescale == 4 or rescale is Trafo.ZERO_UINT8 \
            or (isinstance(rescale, str) and rescale.upper() == Trafo.ZERO_UINT8.name):
        new_image = exposure.rescale_intensity(image,
                                               **dict(dict(in_range=(0, np.nanmax(image)),
                                                           out_range=(0, 255)), **kwargs)
                                               ).astype(np.uint8)

    elif rescale == 5 or rescale is Trafo.EQUALIZE \
            or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE.name):
        norm = HistogramEqualization(**dict(dict(power=1, mask=image > 0), **kwargs))
        new_image = norm(image)

    elif rescale == 6 or rescale is Trafo.EQUALIZE_UINT8 \
            or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE_UINT8.name):
        norm = HistogramEqualization(**dict(dict(power=1, mask=image > 0), **kwargs))
        new_image = np.multiply(norm(image), 255).astype(np.uint8)

    elif rescale == 7 or rescale is Trafo.EQUALIZE_ALL \
            or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE_ALL.name):
        norm = HistogramEqualization(**dict(dict(power=1, mask=None), **kwargs))
        new_image = norm(image)

    elif rescale == 8 or rescale is Trafo.EQUALIZE_ALL_UINT8 \
            or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE_ALL_UINT8.name):
        norm = HistogramEqualization(**dict(dict(power=1, mask=None), **kwargs))
        new_image = np.multiply(norm(image), 255).astype(np.uint8)

    elif rescale == 9 or rescale is Trafo.EQUALIZE_0P3 \
            or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE_0P3.name):
        norm = HistogramEqualization(**dict(dict(power=0.3, mask=image > 0), **kwargs))
        new_image = norm(image)

    elif rescale == 10 or rescale is Trafo.EQUALIZE_0P3_UINT8 \
            or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE_0P3_UINT8.name):
        norm = HistogramEqualization(**dict(dict(power=0.3, mask=image > 0), **kwargs))
        new_image = np.multiply(norm(image), 255).astype(np.uint8)

    elif rescale == 11 or rescale is Trafo.EQUALIZE_0P3_ALL \
            or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE_0P3_ALL.name):
        norm = HistogramEqualization(**dict(dict(power=0.3, mask=None), **kwargs))
        new_image = norm(image)

    elif rescale == 12 or rescale is Trafo.EQUALIZE_0P3_ALL_UINT8 \
            or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE_0P3_ALL_UINT8.name):
        norm = HistogramEqualization(**dict(dict(power=0.3, mask=None), **kwargs))
        new_image = np.multiply(norm(image), 255).astype(np.uint8)

    # to be deprecated eventually
    elif rescale == 'equal':
        new_image = exposure.equalize_hist(image, **kwargs)
    elif rescale == 'unity':
        new_image = exposure.rescale_intensity(image * 1., **kwargs)
    elif isinstance(rescale, tuple):
        p_low, p_high = np.ptp(image) * np.asarray(rescale) / 100 + image.min()
        new_image = exposure.rescale_intensity(image, in_range=(p_low, p_high))
    elif callable(rescale):
        new_image = rescale(image, **kwargs)
    else:
        raise TypeError('Transformation is not defined.')

    return new_image
