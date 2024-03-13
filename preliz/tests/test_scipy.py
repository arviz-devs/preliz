import pytest
from numpy.testing import assert_almost_equal
import numpy as np
from scipy import stats


from preliz.distributions import (
    Beta,
    HalfNormal,
    Normal,
    Bernoulli,
    Binomial,
    NegativeBinomial,
    Poisson,
)


@pytest.mark.parametrize(
    "p_dist, sp_dist, p_params, sp_params",
    [
        (Beta, stats.beta, {"alpha": 2, "beta": 5}, {"a": 2, "b": 5}),
        (Normal, stats.norm, {"mu": 0, "sigma": 2}, {"loc": 0, "scale": 2}),
        (HalfNormal, stats.halfnorm, {"sigma": 2}, {"scale": 2}),
        (Binomial, stats.binom, {"n": 4, "p": 0.4}, {"n": 4, "p": 0.4}),
        (Bernoulli, stats.bernoulli, {"p": 0.4}, {"p": 0.4}),
        (
            NegativeBinomial,
            stats.nbinom,
            {"mu": 3.5, "alpha": 2.1},
            {"n": 2.1, "p": 2.1 / (3.5 + 2.1)},
        ),
        (Poisson, stats.poisson, {"mu": 3.5}, {"mu": 3.5}),
    ],
)
def test_match_scipy(p_dist, sp_dist, p_params, sp_params):
    preliz_dist = p_dist(**p_params)
    scipy_dist = sp_dist(**sp_params)

    actual = preliz_dist.entropy()
    expected = scipy_dist.entropy()
    if preliz_dist.kind == "discrete":
        assert_almost_equal(actual, expected, decimal=1)
    else:
        assert_almost_equal(actual, expected, decimal=4)

    rng = np.random.default_rng(1)
    actual_rvs = preliz_dist.rvs(20, random_state=rng)
    rng = np.random.default_rng(1)
    expected_rvs = scipy_dist.rvs(20, random_state=rng)
    assert_almost_equal(actual_rvs, expected_rvs)

    actual_pdf = preliz_dist.pdf(actual_rvs)
    if preliz_dist.kind == "continuous":
        expected_pdf = scipy_dist.pdf(expected_rvs)
    else:
        expected_pdf = scipy_dist.pmf(expected_rvs)
    assert_almost_equal(actual_pdf, expected_pdf, decimal=4)

    support = preliz_dist.support
    cdf_vals = np.concatenate([actual_rvs, support, [support[0] - 1], [support[1] + 1]])

    actual_cdf = preliz_dist.cdf(cdf_vals)
    expected_cdf = scipy_dist.cdf(cdf_vals)
    assert_almost_equal(actual_cdf, expected_cdf, decimal=6)

    x_vals = [-1, 0, 0.25, 0.5, 0.75, 1, 2]
    actual_ppf = preliz_dist.ppf(x_vals)
    expected_ppf = scipy_dist.ppf(x_vals)
    assert_almost_equal(actual_ppf, expected_ppf)

    actual_logpdf = preliz_dist.logpdf(actual_rvs)
    if preliz_dist.kind == "continuous":
        expected_logpdf = scipy_dist.logpdf(expected_rvs)
    else:
        expected_logpdf = scipy_dist.logpmf(expected_rvs)
    ### Check gammaln implementation
    assert_almost_equal(actual_logpdf, expected_logpdf, decimal=3)

    actual_moments = preliz_dist.moments("mvsk")
    expected_moments = scipy_dist.stats("mvsk")
    assert_almost_equal(actual_moments, expected_moments)
