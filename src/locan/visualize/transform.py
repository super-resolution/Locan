"""
Transform intensity values.

This module provides functions and classes to rescale intensity values.
In addition to the presented functions, rescaling can further be performed by
third-party modules like:
1) matplotlib.colors
2) skimage.exposure
3) astropy.visualization.

Callable transformation classes that inherit from
:class:`matplotlib.colors.Normalize` and specify an inverse transformation
can be passed to the `norm` parameter.

"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
from skimage import exposure

__all__: list[str] = ["Trafo", "HistogramEqualization", "adjust_contrast"]

logger = logging.getLogger(__name__)


class Trafo(Enum):
    """
    Standard definitions for intensity transformation.
    """

    NONE = 0
    """no transformation"""
    STANDARDIZE = 1
    """rescale (min, max) to (0, 1)"""
    STANDARDIZE_UINT8 = 2
    """rescale (min, max) to (0, 255)"""
    ZERO = 3
    """rescale (0, max) to (0, 1)"""
    ZERO_UINT8 = 4
    """rescale (0, max) to (0, 255)"""
    EQUALIZE = 5
    """equalize histogram from all values > 0"""
    EQUALIZE_UINT8 = 6
    """equalize histogram from all values > 0 onto (0, 255)"""
    EQUALIZE_ALL = 7
    """equalize histogram"""
    EQUALIZE_ALL_UINT8 = 8
    """equalize histogram onto (0, 255)"""
    EQUALIZE_0P3 = 9
    """equalize histogram with power = 0.3 from all values > 0"""
    EQUALIZE_0P3_UINT8 = 10
    """equalize histogram with power = 0.3 from all values > 0 onto (0, 255)"""
    EQUALIZE_0P3_ALL = 11
    """equalize histogram with power = 0.3"""
    EQUALIZE_0P3_ALL_UINT8 = 12
    """equalize histogram with power = 0.3 onto (0, 255)"""


class Transform(ABC):
    """
    Abstract base class for transformation classes.
    """

    @abstractmethod
    def __call__(self, values: npt.ArrayLike, clip: bool = True) -> npt.NDArray[Any]:
        """
        Transform values.

        Parameters
        ----------
        values
            The input values
        clip
            If `True` values outside the [0:1] range are
            clipped to the [0:1] range.

        Returns
        -------
        npt.NDArray[Any]
            The transformed values.
        """
        raise NotImplementedError

        # Example implementation:
        # _values, is_scalar = self.process_value(values)
        # self.autoscale_None(_values)  # sets self.vmin, self.vmax if None
        #
        # if clip is None:
        #     clip = self.clip
        #
        # # Normalize based on vmin and vmax
        # np.subtract(_values, self.vmin, out=_values)
        # np.true_divide(_values, self.vmax - self.vmin, out=_values)
        #
        # # Clip to the 0 to 1 range
        # if clip:
        #     np.clip(_values, 0.0, 1.0, out=_values)
        #
        # new_values = _values
        # return new_values

    @abstractmethod
    def inverse(self, values: npt.ArrayLike) -> npt.NDArray[Any]:
        """A transformation that performs the inverse operation."""
        raise NotImplementedError


class HistogramEqualization(mcolors.Normalize, Transform):
    """
    Histogram equalization with power intensification.

    The transformation function as described in [1]_ is :math:`f(a, p)`
    according to:

    .. math::

       \\frac{f(a) - f(a_min)} {f(a_max) - f(a_min)} =
       \\frac{\\int_{a_min}^{a}{h^p(a') da'}} {\\int_{a_min}^{a_max}{h^p(a') da'}}

    Here, :math:`a` is an intensity value, :math:`p` the power (a parameter),
    and :math:`h(a)` the histogram of
    intensities.

    Note
    -----
    The default for n_bins is 65536 (16 bit).
    For most SMLM datasets this should be sufficient to resolve individual
    localizations despite a large dynamic range.
    Setting n_bins to 256 (8 bit) is too course for many SMLM datasets.

    References
    ----------
    .. [1] Yaroslavsky, L. (1985) Digital picture processing. Springer, Berlin.

    Parameters
    ----------
    vmin
        min intensity
    vmax
        max intensity
    reference
        The data values to define the transformation function. If None then
        the values in `__call__` are used.
    power
        The `power` intensification parameter.
    n_bins
        Number of bins used to compute the intensity histogram.
    mask
        A bool mask with shape equal to that of values. If reference is None,
        reference is set to values[mask].
        The transformation determined from reference is then applied to all
        values.
    """

    def __init__(
        self,
        vmin: int | float | None = None,
        vmax: int | float | None = None,
        reference: npt.ArrayLike | None = None,
        power: float = 1,
        n_bins: int = 65536,
        mask: npt.ArrayLike | None = None,
    ) -> None:
        super().__init__(vmin=vmin, vmax=vmax)
        self.reference = reference
        self.power = power
        self.n_bins = n_bins
        self.mask = mask

    def __call__(self, values: npt.ArrayLike) -> npt.NDArray:  # type: ignore
        """
        Histogram equalization with power intensification.

        Parameters
        ----------
        values
            The input values.

        Returns
        -------
        npt.NDArray
        """
        if np.any(np.isnan(values)):
            raise ValueError("HistogramEqualization does not work with nan values.")

        _values, is_scalar = self.process_value(values)
        self.autoscale_None(_values)  # sets self.vmin, self.vmax if None
        assert self.vmin is not None  # type narrowing # noqa: S101
        assert self.vmax is not None  # type narrowing # noqa: S101

        # Normalize based on vmin and vmax
        np.subtract(_values, self.vmin, out=_values)
        np.true_divide(_values, self.vmax - self.vmin, out=_values)

        # Clip to the 0 to 1 range
        np.clip(_values, 0.0, 1.0, out=_values)

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

    def inverse(  # type:ignore[override]
        self, values: npt.ArrayLike
    ) -> npt.NDArray[Any]:
        """A Transformation object that performs the inverse operation."""
        raise NotImplementedError


def adjust_contrast(
    image: npt.ArrayLike,
    rescale: int | str | Trafo | Callable[..., Any] | bool | None = True,
    **kwargs: Any,
) -> npt.NDArray[np.float_]:
    """
    Adjust contrast of image by a predefined transformation:

    Parameters
    ----------
    image
        Values to be adjusted
    rescale
        Transformation as defined in :class:`locan.constants.Trafo` or by
        transformation function.
        For None or False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min
        intensities.
        For 'equal' intensity values are rescaled by histogram equalization.
    kwargs
        Other parameters that are passed to the specific Transformation class.

    Returns
    -------
    npt.NDArray[np.float_]
    """
    image = np.asarray(image)
    if (
        rescale is None
        or rescale is False
        or rescale is Trafo.NONE
        or rescale == 0
        or (isinstance(rescale, str) and rescale.upper() == Trafo.NONE.name)
    ):
        new_image = image

    elif (
        rescale is True
        or rescale == 1
        or rescale is Trafo.STANDARDIZE
        or (isinstance(rescale, str) and rescale.upper() == Trafo.STANDARDIZE.name)
    ):
        new_image = exposure.rescale_intensity(  # type: ignore[no-untyped-call]
            image * 1.0,
            **dict(dict(in_range=(np.nanmin(image), np.nanmax(image))), **kwargs),
        )

    elif (
        rescale == 2
        or rescale is Trafo.STANDARDIZE_UINT8
        or (
            isinstance(rescale, str) and rescale.upper() == Trafo.STANDARDIZE_UINT8.name
        )
    ):
        new_image = exposure.rescale_intensity(  # type: ignore[no-untyped-call]
            image,
            **dict(
                dict(in_range=(np.nanmin(image), np.nanmax(image)), out_range=(0, 255)),
                **kwargs,
            ),
        ).astype(np.uint8)

    elif (
        rescale == 3
        or rescale is Trafo.ZERO
        or (isinstance(rescale, str) and rescale.upper() == Trafo.ZERO.name)
    ):
        new_image = exposure.rescale_intensity(  # type: ignore[no-untyped-call]
            image * 1.0,
            **dict(dict(in_range=(0, np.nanmax(image)), out_range=(0, 1)), **kwargs),
        )

    elif (
        rescale == 4
        or rescale is Trafo.ZERO_UINT8
        or (isinstance(rescale, str) and rescale.upper() == Trafo.ZERO_UINT8.name)
    ):
        new_image = exposure.rescale_intensity(  # type: ignore[no-untyped-call]
            image,
            **dict(dict(in_range=(0, np.nanmax(image)), out_range=(0, 255)), **kwargs),
        ).astype(np.uint8)

    elif (
        rescale == 5
        or rescale is Trafo.EQUALIZE
        or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE.name)
    ):
        norm = HistogramEqualization(**dict(dict(power=1, mask=image > 0), **kwargs))
        new_image = norm(image)

    elif (
        rescale == 6
        or rescale is Trafo.EQUALIZE_UINT8
        or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE_UINT8.name)
    ):
        norm = HistogramEqualization(**dict(dict(power=1, mask=image > 0), **kwargs))
        new_image = np.multiply(norm(image), 255).astype(np.uint8)

    elif (
        rescale == 7
        or rescale is Trafo.EQUALIZE_ALL
        or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE_ALL.name)
    ):
        norm = HistogramEqualization(**dict(dict(power=1, mask=None), **kwargs))
        new_image = norm(image)

    elif (
        rescale == 8
        or rescale is Trafo.EQUALIZE_ALL_UINT8
        or (
            isinstance(rescale, str)
            and rescale.upper() == Trafo.EQUALIZE_ALL_UINT8.name
        )
    ):
        norm = HistogramEqualization(**dict(dict(power=1, mask=None), **kwargs))
        new_image = np.multiply(norm(image), 255).astype(np.uint8)

    elif (
        rescale == 9
        or rescale is Trafo.EQUALIZE_0P3
        or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE_0P3.name)
    ):
        norm = HistogramEqualization(**dict(dict(power=0.3, mask=image > 0), **kwargs))
        new_image = norm(image)

    elif (
        rescale == 10
        or rescale is Trafo.EQUALIZE_0P3_UINT8
        or (
            isinstance(rescale, str)
            and rescale.upper() == Trafo.EQUALIZE_0P3_UINT8.name
        )
    ):
        norm = HistogramEqualization(**dict(dict(power=0.3, mask=image > 0), **kwargs))
        new_image = np.multiply(norm(image), 255).astype(np.uint8)

    elif (
        rescale == 11
        or rescale is Trafo.EQUALIZE_0P3_ALL
        or (isinstance(rescale, str) and rescale.upper() == Trafo.EQUALIZE_0P3_ALL.name)
    ):
        norm = HistogramEqualization(**dict(dict(power=0.3, mask=None), **kwargs))
        new_image = norm(image)

    elif (
        rescale == 12
        or rescale is Trafo.EQUALIZE_0P3_ALL_UINT8
        or (
            isinstance(rescale, str)
            and rescale.upper() == Trafo.EQUALIZE_0P3_ALL_UINT8.name
        )
    ):
        norm = HistogramEqualization(**dict(dict(power=0.3, mask=None), **kwargs))
        new_image = np.multiply(norm(image), 255).astype(np.uint8)

    # to be deprecated eventually
    elif rescale == "equal":
        new_image = exposure.equalize_hist(image, **kwargs)  # type: ignore[no-untyped-call]
    elif rescale == "unity":
        new_image = exposure.rescale_intensity(image * 1.0, **kwargs)  # type: ignore[no-untyped-call]
    elif isinstance(rescale, tuple):
        p_low, p_high = np.ptp(image) * np.asarray(rescale) / 100 + image.min()
        new_image = exposure.rescale_intensity(image, in_range=(p_low, p_high))
    elif callable(rescale):
        new_image = rescale(image, **kwargs)
    else:
        raise TypeError("Transformation is not defined.")

    return new_image
