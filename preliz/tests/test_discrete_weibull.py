from numpy.testing import assert_almost_equal
import numpy as np
from scipy.stats import skew, kurtosis

from preliz.distributions import DiscreteWeibull


def test_disc_weibull_vs_random():
    dist = DiscreteWeibull(0.7, 0.9)

    rng = np.random.default_rng(1)
    rvs = dist.rvs(500_000, random_state=rng)

    assert_almost_equal(dist.mean(), rvs.mean(), decimal=2)
    assert_almost_equal(dist.median(), np.median(rvs), decimal=2)
    assert_almost_equal(dist.var(), rvs.var(), decimal=1)
    assert_almost_equal(dist.std(), rvs.std(), decimal=1)
    assert_almost_equal(dist.skewness(), skew(rvs), decimal=1)
    assert_almost_equal(dist.kurtosis(), kurtosis(rvs), decimal=0)

    q = np.linspace(0.1, 0.9, 10)
    assert_almost_equal(dist.ppf(q), np.quantile(rvs, q), decimal=2)

    for x in np.arange(6):
        assert_almost_equal(dist.pdf(x), np.mean(rvs == x), decimal=2)
        assert_almost_equal(dist.cdf(x), np.sum(dist.pdf(np.arange(x + 1))), decimal=2)
        assert_almost_equal(dist.cdf(x), np.sum(np.exp(dist.logpdf(np.arange(x + 1)))), decimal=2)
        assert_almost_equal(dist.cdf(x), np.mean(rvs <= x), decimal=2)
