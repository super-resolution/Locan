"""

Statistics related tools.

"""
from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

__all__: list[str] = [
    "weighted_mean_variance",
    "ratio_fwhm_to_sigma",
    "biased_variance",
]

logger = logging.getLogger(__name__)


class WeightedMeanVariance(NamedTuple):
    weighted_mean: float | npt.NDArray[np.float_]
    weighted_mean_variance: float | npt.NDArray[np.float_]


def weighted_mean_variance(
    values: npt.ArrayLike, weights: npt.ArrayLike | None
) -> WeightedMeanVariance:
    """
    Compute weighted mean (average)
    and the corresponding weighted mean variance [1]_.

    Parameters
    ----------
    values
        Values from which to compute the weighted average.
    weights
        Weights to use for weighted average.

    Returns
    -------
    WeightedMeanVariance
        weighted_mean, weighted_mean_variance

    References
    ----------
    .. [1] Donald F. Gatz, Luther Smith,
       The standard error of a weighted mean concentration —
       I. Bootstrapping vs other methods.
       Atmospheric Environment 29(11), 1995: 1185-1193,
       https://doi.org/10.1016/1352-2310(94)00210-C.
    """
    values = np.asarray(values)
    if values.ndim > 1:
        raise ValueError("values must be 1-dimensional.")
    if weights is None:
        weights = np.ones(shape=values.shape)
    else:
        weights = np.asarray(weights)
    if values.shape != weights.shape:
        raise TypeError("Shape of values and weights must be the same.")

    n_weights = len(weights)
    if n_weights == 1:
        weighted_average = np.mean(a=values)
        weighted_mean_variance_ = 0
    else:
        weighted_average = np.average(a=values, weights=weights)
        weights_mean = np.mean(a=weights)
        weighted_mean_variance_ = (
            n_weights
            / (n_weights - 1)
            / np.sum(weights) ** 2
            * (
                sum((weights * values - weights_mean * weighted_average) ** 2)
                - 2
                * weighted_average
                * sum(
                    (weights - weights_mean)
                    * (weights * values - weights_mean * weighted_average)
                )
                + weighted_average**2 * sum((weights - weights_mean) ** 2)
            )
        )

        # Check for negative values that should not occur
        # but did result with some floating values
        if (
            weighted_mean_variance_ != 0
            and weighted_mean_variance_[weighted_mean_variance_ < 0].any()  # type: ignore
        ):
            if np.ndim(weighted_mean_variance_) == 0:
                weighted_mean_variance_ = 0
            else:
                weighted_mean_variance_[weighted_mean_variance_ < 0] = 0  # type: ignore
            logger.warning(
                "Negative values for weighted_mean_variance occurred and were set to "
                "zero."
            )

    return WeightedMeanVariance(
        weighted_mean=weighted_average,
        weighted_mean_variance=weighted_mean_variance_,
    )


def ratio_fwhm_to_sigma() -> float:
    """
    The numeric value of the ratio between full-width-half-max (fwhm) width
    and standard deviation (sigma) of a normal distribution.

    Returns
    -------
    float
    """
    return_value = 2 * np.sqrt(2 * np.log(2))
    return float(return_value)


def biased_variance(
    variance: npt.ArrayLike, n_samples: npt.ArrayLike
) -> npt.NDArray[np.float_]:
    """
    The sample variance is biased if not corrected by Bessel's correction.
    This function yields the biased variance by applying the inverse
    correction.

    .. math::

        E(variance_{biased}) = variance * (1 - 1 / localization_counts).

    Parameters
    ----------
    variance
        An unbiased variance.
    n_samples
        Number of samples from which the biased sample variance would be
        computed.

    Returns
    -------
    npt.NDArray[np.float_]
    """
    variance = np.asarray(variance)
    n_samples = np.asarray(n_samples)
    return_value: npt.NDArray[np.float_] = variance * (1 - 1 / n_samples)
    return return_value
