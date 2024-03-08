import pytest
from numpy.testing import assert_almost_equal
import numpy as np
from scipy import stats


from preliz.distributions import Normal, HalfNormal, Poisson


@pytest.mark.parametrize(
    "p_dist, sp_dist, p_params, sp_params",
    [
        (Normal, stats.norm, {"mu": 0, "sigma": 2}, {"loc": 0, "scale": 2}),
        (HalfNormal, stats.halfnorm, {"sigma": 2}, {"scale": 2}),
        (Poisson, stats.poisson, {"mu": 3.5}, {"mu": 3.5}),
    ],
)
def test_match_scipy(p_dist, sp_dist, p_params, sp_params):
    preliz_dist = p_dist(**p_params)
    scipy_dist = sp_dist(**sp_params)

    actual = preliz_dist.entropy()
    expected = scipy_dist.entropy()
    if preliz_dist.kind == "discrete":
        assert_almost_equal(actual, expected, decimal=2)
    else:
        assert_almost_equal(actual, expected)

    rng = np.random.default_rng(1)
    actual_rvs = preliz_dist.rvs(100, random_state=rng)
    rng = np.random.default_rng(1)
    expected_rvs = scipy_dist.rvs(100, random_state=rng)
    assert_almost_equal(actual_rvs, expected_rvs)

    actual_pdf = preliz_dist.pdf(actual_rvs)
    if preliz_dist.kind == "continuous":
        expected_pdf = scipy_dist.pdf(expected_rvs)
    else:
        expected_pdf = scipy_dist.pmf(expected_rvs)
    assert_almost_equal(actual_pdf, expected_pdf)

    actual_cdf = preliz_dist.cdf(actual_rvs)
    expected_cdf = scipy_dist.cdf(expected_rvs)
    assert_almost_equal(actual_cdf, expected_cdf)

    x_vals = np.linspace(0, 1, 10)
    actual_ppf = preliz_dist.ppf(x_vals)
    expected_ppf = scipy_dist.ppf(x_vals)
    assert_almost_equal(actual_ppf, expected_ppf)

    actual_logpdf = preliz_dist.logpdf(actual_rvs)
    if preliz_dist.kind == "continuous":
        expected_logpdf = scipy_dist.logpdf(expected_rvs)
    else:
        expected_logpdf = scipy_dist.logpmf(expected_rvs)
    assert_almost_equal(actual_logpdf, expected_logpdf)

    actual_moments = preliz_dist.moments("mvsk")
    expected_moments = scipy_dist.stats("mvsk")
    assert_almost_equal(actual_moments, expected_moments)
