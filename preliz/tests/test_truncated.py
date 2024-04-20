from numpy.testing import assert_almost_equal
import numpy as np

from preliz.distributions import Truncated, TruncatedNormal, Normal


def test_truncated():
    custom_truncnorm_dist = Truncated(Normal(0, 2), -1, np.inf)
    genera_truncnorm_dist = TruncatedNormal(0, 2, -1, np.inf)

    rng = np.random.default_rng(1)
    actual_rvs = custom_truncnorm_dist.rvs(20, random_state=rng)
    rng = np.random.default_rng(1)
    expected_rvs = genera_truncnorm_dist.rvs(20, random_state=rng)
    assert_almost_equal(actual_rvs, expected_rvs)

    actual_pdf = custom_truncnorm_dist.pdf(actual_rvs)
    expected_pdf = genera_truncnorm_dist.pdf(actual_rvs)
    assert_almost_equal(actual_pdf, expected_pdf, decimal=4)

    support = custom_truncnorm_dist.support
    cdf_vals = np.concatenate([actual_rvs, support, [support[0] - 1], [support[1] + 1]])

    actual_cdf = custom_truncnorm_dist.cdf(cdf_vals)
    expected_cdf = genera_truncnorm_dist.cdf(cdf_vals)
    assert_almost_equal(actual_cdf, expected_cdf, decimal=6)

    x_vals = [-1, 0, 0.25, 0.5, 0.75, 1, 2]
    actual_ppf = custom_truncnorm_dist.ppf(x_vals)
    expected_ppf = genera_truncnorm_dist.ppf(x_vals)
    assert_almost_equal(actual_ppf, expected_ppf)

    actual_logpdf = custom_truncnorm_dist.logpdf(actual_rvs)
    expected_logpdf = genera_truncnorm_dist.logpdf(actual_rvs)
    assert_almost_equal(actual_logpdf, expected_logpdf)

    actual_neg_logpdf = custom_truncnorm_dist._neg_logpdf(actual_rvs)
    expected_neg_logpdf = -expected_logpdf.sum()
    assert_almost_equal(actual_neg_logpdf, expected_neg_logpdf)

    actual_median = custom_truncnorm_dist.median()
    expected_median = genera_truncnorm_dist.median()
    assert_almost_equal(actual_median, expected_median)

    actual_mean = custom_truncnorm_dist.mean()
    expected_mean = genera_truncnorm_dist.mean()
    assert_almost_equal(actual_mean, expected_mean, decimal=2)

    actual_var = custom_truncnorm_dist.var()
    expected_var = genera_truncnorm_dist.var()
    assert_almost_equal(actual_var, expected_var, decimal=2)

    actual_skew = custom_truncnorm_dist.skewness()
    expected_skew = genera_truncnorm_dist.skewness()
    assert_almost_equal(actual_skew, expected_skew, decimal=2)

    actual_kurt = custom_truncnorm_dist.kurtosis()
    expected_kurt = genera_truncnorm_dist.kurtosis()
    assert_almost_equal(actual_kurt, expected_kurt, decimal=1)

    actual_entropy = custom_truncnorm_dist.entropy()
    expected_entropy = genera_truncnorm_dist.entropy()
    assert_almost_equal(actual_entropy, expected_entropy, decimal=2)
