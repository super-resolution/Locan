"""

Statistics related tools.

"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["weighted_mean_variance", "ratio_fwhm_to_sigma"]


class WeightedMeanVariance(NamedTuple):
    weighted_mean: np.array
    weighted_mean_variance: np.array


def weighted_mean_variance(values, weights) -> WeightedMeanVariance:
    """
    Compute weighted mean (average)
    and the corresponding weighted mean variance.

    Parameters
    ----------
    values : array-like
        Values from which to compute the weighted average.
    weights : array-like | None
        Weights to use for weighted average.

    # todo Reference

    Returns
    -------
    WeightedMeanVariance
        weighted_mean, weighted_mean_variance
    """
    values = np.asarray(values)
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

    return WeightedMeanVariance(
        weighted_mean=weighted_average, weighted_mean_variance=weighted_mean_variance_
    )


def ratio_fwhm_to_sigma() -> float:
    """
    The numeric value of the ratio between full-width-half-max (fwhm) width
    and standard deviation (sigma) of a normal distribution.

    Returns
    -------
    float
    """
    return 2 * np.sqrt(2 * np.log(2))
