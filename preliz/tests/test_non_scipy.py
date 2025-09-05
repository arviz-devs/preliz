import numpy as np
from numpy.testing import assert_almost_equal
from scipy import integrate
from scipy.stats import kurtosis, skew

from preliz.distributions import Categorical, DiscreteWeibull, ScaledInverseChiSquared


def test_categorical():
    p = [0.2, 0.5, 0.3]
    q = [0.2, 0.7, 1]
    dist = Categorical(p)
    assert dist.p.sum() == 1.0
    assert all(dist.pdf([0, 1, 2]) == p)
    assert all(dist.cdf([0, 1, 2]) == np.cumsum(p))
    assert all(dist.cdf(dist.ppf(q)) == q)


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


def test_scaled_inverse_chi2():
    dist = ScaledInverseChiSquared(15, 1)

    rng = np.random.default_rng(1)
    rvs = dist.rvs(500_000, random_state=rng)

    assert_almost_equal(dist.mean(), rvs.mean(), decimal=2)
    assert_almost_equal(dist.median(), np.median(rvs), decimal=2)
    assert_almost_equal(dist.var(), rvs.var(), decimal=1)
    assert_almost_equal(dist.std(), rvs.std(), decimal=1)
    assert_almost_equal(dist.skewness(), skew(rvs), decimal=0)
    assert_almost_equal(dist.kurtosis(), kurtosis(rvs), decimal=0)

    q = np.linspace(0.1, 0.9, 10)
    assert_almost_equal(dist.ppf(q), np.quantile(rvs, q), decimal=2)
    assert_almost_equal(dist.cdf(dist.ppf(q)), q, decimal=2)

    for x in np.arange(6):
        assert_almost_equal(dist.cdf(x), np.mean(rvs <= x), decimal=2)

    assert_almost_equal(integrate.quad(dist.pdf, 0.1, 50)[0], 1, decimal=5)

    for x in np.linspace(0.2, 10, 5):
        val, _ = integrate.quad(dist.pdf, 0, x)
        assert_almost_equal(val, dist.cdf(x))
