import pytest
from numpy.testing import assert_almost_equal
import numpy as np

from preliz.distributions import (
    AsymmetricLaplace,
    Beta,
    Cauchy,
    ChiSquared,
    Gamma,
    Gumbel,
    ExGaussian,
    Exponential,
    HalfCauchy,
    HalfNormal,
    HalfStudent,
    InverseGamma,
    Laplace,
    Logistic,
    LogNormal,
    Moyal,
    Normal,
    Pareto,
    SkewNormal,
    Student,
    Triangular,
    TruncatedNormal,
    Uniform,
    VonMises,
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
        (AsymmetricLaplace, (1, 0, 1)),
        (Beta, (2, 5)),
        (ChiSquared, (1,)),
        (ExGaussian, (0, 1, 1e-6)),
        (Exponential, (0.5,)),
        (Gamma, (1, 0.5)),
        (Gumbel, (1, 2)),
        (HalfNormal, (1,)),
        (HalfStudent, (100, 1)),
        (InverseGamma, (3, 1)),
        (Laplace, (0, 1)),
        (Logistic, (1, 2)),
        (LogNormal, (0, 0.5)),
        (Moyal, (1, 2)),
        (Normal, (0, 1)),
        (Pareto, (5, 1)),
        (SkewNormal, (0, 1, 0)),
        (Student, (100, 0, 1)),
        (Triangular, (-2, 3, 7)),
        (TruncatedNormal, (0, 1, -np.inf, np.inf)),
        (Uniform, (0, 1)),
        (VonMises, (0, 1000)),
        (Wald, (1, 1)),
        (Weibull, (2, 1)),
        (Binomial, (2, 0.5)),
        (Binomial, (2, 0.1)),
        (NegativeBinomial, (8, 4)),
        (Poisson, (4.5,)),
        (DiscreteUniform, (0, 1)),
    ],
)
def test_moments(distribution, params):
    dist = distribution(*params)
    dist_ = distribution()

    dist_._fit_moments(dist.rv_frozen.mean(), dist.rv_frozen.std())

    tol = 5
    if dist.name in ["binomial", "student"]:
        tol = 0
    assert_almost_equal(dist.rv_frozen.mean(), dist_.rv_frozen.mean(), tol)
    assert_almost_equal(dist.rv_frozen.std(), dist_.rv_frozen.std(), tol)
    assert_almost_equal(params, dist_.params, 0)


@pytest.mark.parametrize(
    "distribution, params",
    [
        (AsymmetricLaplace, (1, 4, 3)),
        (Beta, (2, 5)),
        (Cauchy, (0, 1)),
        (ChiSquared, (1,)),
        (ExGaussian, (0, 1, 3)),
        (Exponential, (0.5,)),
        (Gamma, (1, 0.5)),
        (Gumbel, (0, 1)),
        (HalfCauchy, (1,)),
        (HalfNormal, (1,)),
        (HalfStudent, (100, 1)),
        (InverseGamma, (3, 0.5)),
        (Laplace, (0, 1)),
        (Logistic, (0, 1)),
        (LogNormal, (0, 0.5)),
        (Moyal, (0, 2)),
        (Normal, (0, 1)),
        (Pareto, (5, 1)),
        (SkewNormal, (0, 1, 0)),
        (SkewNormal, (0, 1, -1)),
        (Student, (4, 0, 1)),
        (Student, (1000, 0, 1)),
        (Triangular, (-3, 0, 5)),
        (TruncatedNormal, (0, 1, -1, 1)),
        (Uniform, (0, 1)),
        (VonMises, (0, 1)),
        (Wald, (1, 1)),
        (Weibull, (2, 1)),
        (Binomial, (2, 0.5)),
        (Binomial, (2, 0.1)),
        (NegativeBinomial, (8, 4)),
        (Poisson, (4.5,)),
        (DiscreteUniform, (0, 1)),
    ],
)
def test_mle(distribution, params):
    dist = distribution(*params)
    sample = dist.rv_frozen.rvs(20000)
    dist_ = distribution()
    dist_._fit_mle(sample)

    if dist.name == "pareto":
        tol = 0
    else:
        tol = 1
    assert_almost_equal(dist.rv_frozen.mean(), dist_.rv_frozen.mean(), tol)
    assert_almost_equal(dist.rv_frozen.std(), dist_.rv_frozen.std(), tol)
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


# @pytest.mark.parametrize(
#     "distribution, params, alt_names",
#     [
#         (Beta, (2, 5), ("mu", "sigma")),
#         (Beta, (5, 2), ("mu", "kappa")),
#         (Gamma, (2, 1), ("mu", "sigma")),
#         (HalfNormal, (1,), ("tau",)),
#         (HalfStudent, (1000, 1), ("nu", "lam")),
#         (InverseGamma, (0, 2), ("mu", "sigma")),
#         (Normal, (0, 1), ("mu", "tau")),
#         (SkewNormal, (0, 1, 0), ("mu", "tau", "alpha")),
#         (Student, (1000, 0, 1), ("nu", "mu", "lam")),
#     ],
# )
# def test_alternative_parametrization(distribution, params, alt_names):
#     dist0 = distribution(*params)
#     params1 = {p: getattr(dist0, p) for p in alt_names}
#     dist1 = distribution(**params1)

#     assert_almost_equal(dist0.params, dist1.params)
#     for p in alt_names:
#         assert p in dist1.__repr__()
