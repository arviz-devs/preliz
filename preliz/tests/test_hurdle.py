import pytest
from numpy.testing import assert_almost_equal
import numpy as np

from preliz.distributions import (
    Hurdle,
    Truncated,
    Gamma,
    Normal,
    LogNormal,
    Poisson,
    NegativeBinomial,
)
from preliz.internal.distribution_helper import eps


@pytest.mark.parametrize(
    "dist",
    [
        (Gamma(3, 5)),
        (LogNormal(0, 0.5)),
        (Normal(0, 2)),
        (Poisson(3.5)),
        (NegativeBinomial(3, 5)),
    ],
)
def test_hurdle_vs_truncated(dist):
    if dist.kind == "discrete":
        lower = 1
    else:
        lower = eps

    hurdle_dist = Hurdle(dist, 1)
    base_dist = Truncated(dist, lower=lower, upper=np.inf)

    rng = np.random.default_rng(1)
    actual_rvs = hurdle_dist.rvs(100_000, random_state=rng)
    expected_rvs = base_dist.rvs(100_000, random_state=rng)
    assert_almost_equal(actual_rvs.mean(), expected_rvs.mean(), decimal=2)
    assert_almost_equal(actual_rvs.var(), expected_rvs.var(), decimal=2)

    assert_almost_equal(hurdle_dist.mean(), base_dist.mean(), decimal=2)
    assert_almost_equal(hurdle_dist.var(), base_dist.var(), decimal=3)
    assert_almost_equal(hurdle_dist.std(), base_dist.std(), decimal=4)
    assert_almost_equal(hurdle_dist.entropy(), base_dist.entropy())

    few_rvs = hurdle_dist.rvs(20, random_state=rng)
    assert_almost_equal(hurdle_dist.pdf(few_rvs), base_dist.pdf(few_rvs))
    assert_almost_equal(hurdle_dist.cdf(few_rvs), base_dist.cdf(few_rvs))
    assert_almost_equal(hurdle_dist.logpdf(few_rvs), base_dist.logpdf(few_rvs))
    assert_almost_equal(hurdle_dist._neg_logpdf(few_rvs), base_dist._neg_logpdf(few_rvs))
    x_vals = [-1, 0, 0.25, 0.5, 0.75, 1, 2]
    assert_almost_equal(hurdle_dist.ppf(x_vals), base_dist.ppf(x_vals))


@pytest.mark.parametrize(
    "dist",
    [
        (Gamma(3, 5)),
        (LogNormal(0, 0.5)),
        (Normal(0, 2)),
        (Poisson(3.5)),
        (NegativeBinomial(3, 5)),
    ],
)
def test_hurdle_vs_random(dist):
    hurdle_dist = Hurdle(dist, psi=0.7)

    rng = np.random.default_rng(1)
    rvs = hurdle_dist.rvs(500_000, random_state=rng)

    assert_almost_equal(hurdle_dist.mean(), rvs.mean(), decimal=2)
    assert_almost_equal(hurdle_dist.var(), rvs.var(), decimal=0)
    assert_almost_equal(hurdle_dist.std(), rvs.std(), decimal=0)
