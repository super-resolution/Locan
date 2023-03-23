import pytest

from locan.utils.statistics import WeightedMeanVariance, weighted_mean_variance


def test_weighted_mean_variance():
    wmv = weighted_mean_variance(values=[1, 2], weights=[1, 1])
    assert wmv == WeightedMeanVariance(weighted_mean=1.5, weighted_mean_variance=0.25)

    wmv = weighted_mean_variance(values=[1, 2], weights=None)
    assert wmv == WeightedMeanVariance(weighted_mean=1.5, weighted_mean_variance=0.25)

    with pytest.raises(TypeError):
        weighted_mean_variance(values=[1, 2], weights=[1])

    wmv = weighted_mean_variance(values=[1, 2], weights=[1, 2])
    assert wmv == pytest.approx((1.667, 0.198), rel=0.01)

    wmv = weighted_mean_variance(values=[1], weights=[2])
    assert wmv == pytest.approx((1, 0), rel=0.01)