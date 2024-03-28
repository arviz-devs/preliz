import pytest
from numpy.testing import assert_almost_equal
import numpy as np
from scipy.stats import skew, kurtosis

from preliz.distributions import Censored, Normal, Poisson


@pytest.mark.parametrize(
    "dist, lower, upper",
    [
        (Normal(0, 2), -2, 2),
        (Poisson(3.5), 1, 6),
    ],
)
def test_censored(dist, lower, upper):
    cen_dist = Censored(dist, lower, upper)
    cen_dist_inf = Censored(dist, -np.inf, np.inf)

    x_vals = cen_dist.rvs(1000000, random_state=1)
    assert_almost_equal(np.mean(x_vals == lower), dist.cdf(lower), decimal=2)
    if dist.kind == "discrete":
        assert_almost_equal(np.mean(x_vals == upper), 1 - dist.cdf(upper - 1), decimal=2)
    else:
        assert_almost_equal(np.mean(x_vals == upper), 1 - dist.cdf(upper), decimal=2)

    x_inside = x_vals[(x_vals > lower) & (x_vals < upper)]
    assert_almost_equal(dist.logpdf(x_inside), cen_dist.logpdf(x_inside))
    assert_almost_equal(dist.cdf(x_inside), cen_dist.cdf(x_inside))

    assert_almost_equal(dist.cdf(x_inside), cen_dist.cdf(x_inside))

    assert_almost_equal(cen_dist.median(), dist.median())
    assert_almost_equal(x_vals.mean(), cen_dist.mean(), decimal=1)
    assert_almost_equal(x_vals.var(), cen_dist.var(), decimal=1)
    assert_almost_equal(skew(x_vals), cen_dist.skewness(), decimal=0)
    assert_almost_equal(kurtosis(x_vals), cen_dist.kurtosis(), decimal=0)

    actual_mean = dist.mean()
    expected_mean = cen_dist_inf.mean()
    assert_almost_equal(actual_mean, expected_mean, decimal=2)

    actual_var = dist.var()
    expected_var = cen_dist_inf.var()
    assert_almost_equal(actual_var, expected_var, decimal=2)

    actual_entropy = dist.entropy()
    expected_entropy = cen_dist_inf.entropy()
    assert_almost_equal(actual_entropy, expected_entropy, decimal=1)

    c_l, c_u = cen_dist.hdi()
    d_l, d_u = dist.hdi()
    assert c_l >= d_l
    assert c_u <= d_u
    assert_almost_equal(cen_dist_inf.hdi(), dist.hdi())
