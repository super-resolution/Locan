import numpy as np
import pytest

from locan.utils.statistics import (
    WeightedMeanVariance,
    ratio_fwhm_to_sigma,
    weighted_mean_variance,
)


def test_weighted_mean_variance():
    wmv = weighted_mean_variance(values=[1, 2], weights=[1, 1])
    assert wmv == WeightedMeanVariance(weighted_mean=1.5, weighted_mean_variance=0.25)

    wmv = weighted_mean_variance(values=[1, 2], weights=None)
    assert wmv == WeightedMeanVariance(weighted_mean=1.5, weighted_mean_variance=0.25)

    wmv = weighted_mean_variance(values=[1, 2], weights=[1, 0])
    assert wmv == WeightedMeanVariance(weighted_mean=1, weighted_mean_variance=0.0)

    wmv = weighted_mean_variance(values=[1, 2], weights=[1, -2])
    assert wmv == WeightedMeanVariance(weighted_mean=3, weighted_mean_variance=16.0)

    wmv = weighted_mean_variance(values=[1, 2], weights=[1, 1e100])
    assert wmv == pytest.approx((2, 0), rel=1e-6)

    wmv = weighted_mean_variance(values=[1, 2], weights=[1, np.nan])
    assert np.isnan([wmv.weighted_mean, wmv.weighted_mean_variance]).all

    with pytest.raises(TypeError):
        weighted_mean_variance(values=[1, 2], weights=[1])

    wmv = weighted_mean_variance(values=[1, 2], weights=[1, 2])
    assert wmv == pytest.approx((1.667, 0.198), rel=0.01)

    wmv = weighted_mean_variance(values=[1], weights=[2])
    assert wmv == pytest.approx((1, 0), rel=0.01)


def test_ratio_fwhm_to_sigma():
    assert ratio_fwhm_to_sigma() == pytest.approx(2.355, rel=0.001)
