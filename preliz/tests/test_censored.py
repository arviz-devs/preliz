import pytest
from numpy.testing import assert_almost_equal
import numpy as np

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

    x_vals = cen_dist.rvs(10000, random_state=1)
    assert_almost_equal(x_vals.mean(), dist.mean(), decimal=1)
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

    c_l, c_u = cen_dist.hdi()
    d_l, d_u = dist.hdi()
    assert c_l >= d_l
    assert c_u <= d_u
    assert_almost_equal(cen_dist_inf.hdi(), dist.hdi())
