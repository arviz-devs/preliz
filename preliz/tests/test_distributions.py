import pytest
from numpy.testing import assert_almost_equal
import numpy as np

from preliz.distributions import (
    Beta,
    Cauchy,
    Gamma,
    Exponential,
    HalfCauchy,
    HalfNormal,
    HalfStudent,
    InverseGamma,
    Laplace,
    LogNormal,
    Normal,
    Pareto,
    SkewNormal,
    Student,
    TruncatedNormal,
    Uniform,
    Wald,
    Weibull,
    Binomial,
    DiscreteUniform,
    NegativeBinomial,
    Poisson,
)


@pytest.mark.parametrize(
    "distribution, params",
    [
        (Normal, (0, 1)),
        (Beta, (2, 5)),
        (Gamma, (1, 0.5)),
        (HalfNormal, (1,)),
        (HalfStudent, (3, 1)),
        (HalfStudent, (1000000, 1)),
        (InverseGamma, (3, 1)),
        (Laplace, (0, 1)),
        (LogNormal, (0, 0.5)),
        (Exponential, (0.5,)),
        (Pareto, (5, 1)),
        (SkewNormal, (0, 1, 0)),
        (Student, (3, 0, 1)),
        (Student, (1000000, 0, 1)),
        (TruncatedNormal, (0, 1, -np.inf, np.inf)),
        (Uniform, (0, 1)),
        (Wald, (1, 1)),
        (Weibull, (2, 1)),
        (Binomial, (2, 0.5)),
        (Binomial, (2, 0.1)),
        (NegativeBinomial, (2, 0.7)),
        (NegativeBinomial, (2, 0.3)),
        (Poisson, (4.5,)),
        (DiscreteUniform, (0, 1)),
    ],
)
def test_moments(distribution, params):
    dist = distribution(*params)
    if "student" in dist.name:
        dist_ = distribution(nu=params[0])
    else:
        dist_ = distribution()

    dist_._fit_moments(dist.rv_frozen.mean(), dist.rv_frozen.std())

    tol = 5
    if dist.name == "binomial":
        tol = 0
    assert_almost_equal(dist.rv_frozen.mean(), dist_.rv_frozen.mean(), tol)
    assert_almost_equal(dist.rv_frozen.std(), dist_.rv_frozen.std(), tol)
    assert_almost_equal(params, dist_.params, 0)


@pytest.mark.parametrize(
    "distribution, params",
    [
        (Normal, (0, 1)),
        (Beta, (2, 5)),
        (Cauchy, (0, 1)),
        (Gamma, (1, 0.5)),
        (HalfCauchy, (1,)),
        (HalfNormal, (1,)),
        (HalfStudent, (3, 1)),
        (InverseGamma, (3, 0.5)),
        (Laplace, (0, 1)),
        (LogNormal, (0, 0.5)),
        (Exponential, (0.5,)),
        (Pareto, (5, 1)),
        (SkewNormal, (0, 1, 0)),
        (SkewNormal, (0, 1, -1)),
        (Student, (4, 0, 1)),
        (Student, (1000, 0, 1)),
        (TruncatedNormal, (0, 1, -1, 1)),
        (Uniform, (0, 1)),
        (Wald, (1, 1)),
        (Weibull, (2, 1)),
        (Binomial, (2, 0.5)),
        (Binomial, (2, 0.1)),
        (NegativeBinomial, (2, 0.7)),
        (NegativeBinomial, (2, 0.3)),
        (Poisson, (4.5,)),
        (DiscreteUniform, (0, 1)),
    ],
)
def test_mle(distribution, params):
    dist = distribution(*params)
    sample = dist.rv_frozen.rvs(20000)
    dist_ = distribution()
    dist_._fit_mle(sample)

    assert_almost_equal(dist.rv_frozen.mean(), dist_.rv_frozen.mean(), 1)
    assert_almost_equal(dist.rv_frozen.std(), dist_.rv_frozen.std(), 1)
    if dist.name == "student":
        assert_almost_equal(params[1:], dist_.params[1:], 0)
    else:
        assert_almost_equal(params, dist_.params, 0)


@pytest.mark.parametrize("fmt", (".2f", ".1g"))
@pytest.mark.parametrize("mass", (0.5, 0.95))
def test_summary(fmt, mass):
    result = Normal(0, 1).summary(fmt, mass)
    assert result.mean == 0
    assert result._fields == ("mean", "median", "std", "lower", "upper")
    result = Poisson(2).summary()
    assert result.mean == 2
