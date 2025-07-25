import numpy as np
import pytest
from numpy.testing import assert_almost_equal

try:
    from pymc import Model
except ImportError:
    pass
from preliz.distributions import (
    AsymmetricLaplace,
    Bernoulli,
    Beta,
    BetaBinomial,
    BetaScaled,
    Binomial,
    Cauchy,
    Censored,
    ChiSquared,
    Dirichlet,
    DiscreteUniform,
    DiscreteWeibull,
    ExGaussian,
    Exponential,
    Gamma,
    Geometric,
    Gumbel,
    HalfCauchy,
    HalfNormal,
    HalfStudentT,
    Hurdle,
    InverseGamma,
    Kumaraswamy,
    Laplace,
    Logistic,
    LogitNormal,
    LogLogistic,
    LogNormal,
    Moyal,
    MvNormal,
    NegativeBinomial,
    Normal,
    Pareto,
    Poisson,
    Rice,
    SkewNormal,
    StudentT,
    Triangular,
    Truncated,
    TruncatedNormal,
    Uniform,
    VonMises,
    Wald,
    Weibull,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)


@pytest.fixture(scope="session")
def a_few_poissons():
    return Poisson(), Poisson([1.0, 2.0]), Poisson(4.5)


@pytest.fixture(scope="session")
def multivariates():
    return Dirichlet(), Dirichlet([1.0, 2.0, 1.0]), MvNormal(np.zeros(2), np.eye(2))


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
        (HalfStudentT, (100, 1)),
        (InverseGamma, (3, 1)),
        (Kumaraswamy, (5, 2)),
        (Laplace, (0, 1)),
        (Logistic, (1, 2)),
        (LogLogistic, (1, 5)),
        (LogNormal, (0, 0.5)),
        (LogitNormal, (0, 0.5)),
        (Moyal, (1, 2)),
        (Normal, (0, 1)),
        (Pareto, (5, 1)),
        (Rice, (4, 1)),
        (SkewNormal, (0, 1, 0)),
        (StudentT, (100, 0, 1)),
        (Triangular, (-2, 3, 7)),
        (TruncatedNormal, (0, 1, -np.inf, np.inf)),
        (Uniform, (0, 1)),
        (VonMises, (0, 1000)),
        (Wald, (1, 1)),
        (Weibull, (2, 1)),
        (Bernoulli, (0.8,)),
        (BetaBinomial, (1, 1, 10)),
        (Binomial, (2, 0.5)),
        (Binomial, (2, 0.1)),
        (DiscreteUniform, (0, 1)),
        (DiscreteWeibull, (0.5, 3)),
        (Geometric, (0.75,)),
        (NegativeBinomial, (8, 4)),
        (Poisson, (4.5,)),
        (
            ZeroInflatedPoisson,
            (
                0.8,
                4.5,
            ),
        ),
    ],
)
def test_moments(distribution, params):
    dist = distribution(*params)
    dist_ = distribution()

    dist_._fit_moments(dist.mean(), dist.std())

    tol = 5
    if dist.__class__.__name__ in [
        "BetaBinomial",
        "Binomial",
        "DiscreteWeibull",
        "ExGaussian",
        "NegativeBinomial",
        "Kumaraswamy",
        "LogitNormal",
        "Rice",
        "StudentT",
    ]:
        tol = 0
    assert_almost_equal(dist.mean(), dist_.mean(), tol)
    assert_almost_equal(dist.std(), dist_.std(), tol)
    assert_almost_equal(params, dist_.params, 0)


@pytest.mark.parametrize(
    "distribution, params",
    [
        (AsymmetricLaplace, (1, 4, 3)),
        (Beta, (2, 5)),
        (BetaScaled, (2, 2, -1, 2)),
        (Cauchy, (0, 1)),
        (ChiSquared, (1,)),
        (ExGaussian, (0, 1, 3)),
        (Exponential, (0.5,)),
        (Gamma, (1, 0.5)),
        (Gumbel, (0, 1)),
        (HalfCauchy, (1,)),
        (HalfNormal, (1,)),
        (HalfStudentT, (100, 1)),
        (InverseGamma, (3, 0.5)),
        (Kumaraswamy, (2, 3)),
        (Laplace, (0, 1)),
        (Logistic, (0, 1)),
        (LogNormal, (0, 0.5)),
        (LogitNormal, (0, 0.5)),
        (Moyal, (0, 2)),
        (Normal, (0, 1)),
        (Pareto, (5, 1)),
        (Rice, (1, 2)),
        (SkewNormal, (0, 1, 0)),
        (SkewNormal, (0, 1, -1)),
        (StudentT, (4, 0, 1)),
        (StudentT, (1000, 0, 1)),
        (Triangular, (-3, 0, 5)),
        (TruncatedNormal, (0, 1, -1, 1)),
        (Uniform, (0, 1)),
        (VonMises, (0, 1)),
        (Wald, (1, 1)),
        (Weibull, (2, 1)),
        (Bernoulli, (0.4,)),
        (BetaBinomial, (2, 2, 10)),
        (Binomial, (2, 0.5)),
        (Binomial, (2, 0.1)),
        (DiscreteUniform, (0, 1)),
        (DiscreteWeibull, (0.1, 1.5)),
        (Geometric, (0.5,)),
        (NegativeBinomial, (8, 4)),
        (Poisson, (4.5,)),
        (ZeroInflatedNegativeBinomial, (0.7, 8, 4)),
        (
            ZeroInflatedPoisson,
            (
                0.8,
                4.5,
            ),
        ),
    ],
)
def test_mle(distribution, params):
    for _ in range(10):
        dist = distribution(*params)
        sample = dist.rvs(20_000)
        dist_ = distribution()
        dist_._fit_mle(sample)

        if dist.__class__.__name__ in ["Pareto", "ExGaussian"]:
            tol = 0
        else:
            tol = 1
        try:
            assert_almost_equal(dist.mean(), dist_.mean(), tol)
            assert_almost_equal(dist.std(), dist_.std(), tol)
            if dist.__class__.__name__ == "StudentT":
                assert_almost_equal(params[1:], dist_.params[1:], 0)
            else:
                assert_almost_equal(params, dist_.params, 0)
            break
        except AssertionError:
            pass
    else:
        raise AssertionError("Test failed after 10 attempts")


@pytest.mark.parametrize(
    "distribution, params",
    [
        (AsymmetricLaplace, (1, 4, 3)),
        (Beta, (2, 5)),
        (BetaScaled, (2, 2, -1, 2)),
        (Cauchy, (0, 1)),
        (ChiSquared, (1,)),
        (ExGaussian, (0, 1, 3)),
        (Exponential, (0.5,)),
        (Gamma, (1, 0.5)),
        (Gumbel, (0, 1)),
        (HalfCauchy, (1,)),
        (HalfNormal, (1,)),
        (HalfStudentT, (100, 1)),
        (InverseGamma, (3, 0.5)),
        (Kumaraswamy, (2, 3)),
        (Laplace, (0, 1)),
        (Logistic, (0, 1)),
        (LogNormal, (0, 0.5)),
        (LogitNormal, (0, 0.5)),
        (Moyal, (0, 2)),
        (Normal, (0, 1)),
        (Pareto, (5, 1)),
        (Rice, (1, 2)),
        (SkewNormal, (0, 1, 0)),
        (SkewNormal, (0, 1, -1)),
        (StudentT, (4, 0, 1)),
        (StudentT, (1000, 0, 1)),
        (Triangular, (-3, 0, 5)),
        (TruncatedNormal, (0, 1, -1, 1)),
        (Uniform, (0, 1)),
        (VonMises, (0, 1)),
        (Wald, (1, 1)),
        (Weibull, (2, 1)),
        (BetaBinomial, (2, 2, 10)),
        (Binomial, (20, 0.5)),
        (DiscreteUniform, (-10, 10)),
        (DiscreteWeibull, (0.5, 1)),
        (Geometric, (0.5,)),
        (NegativeBinomial, (8, 4)),
        (Poisson, (4.5,)),
        (ZeroInflatedNegativeBinomial, (0.7, 8, 4)),
        (
            ZeroInflatedPoisson,
            (
                0.8,
                4.5,
            ),
        ),
    ],
)
def test_hdi(distribution, params):
    dist = distribution(*params)
    sample = dist.rvs(10_000)
    interval = dist.hdi(mass=0.8)
    assert_almost_equal(np.mean((sample >= interval[0]) & (sample < interval[1])), 0.8, decimal=1)


@pytest.mark.parametrize("fmt", (".2f", ".1g"))
@pytest.mark.parametrize("interval", ("hdi", "eti", [0.25, 0.75]))
@pytest.mark.parametrize("mass", (0.5, 0.95))
def test_summary_args(mass, interval, fmt):
    result = Normal(0, 1).summary(mass=mass, interval=interval, fmt=fmt)
    assert result.mean == 0
    assert result.std == 1


def test_summary_univariate_valid(a_few_poissons):
    d_0, d_1, d_2 = a_few_poissons
    with pytest.raises(ValueError):
        d_0.summary()
    with pytest.raises(ValueError):
        d_1.summary()
    result = d_2.summary()
    assert result.__class__.__name__ == "Poisson"
    assert result.mean == 4.5
    assert result.std == 2.12
    assert result.lower == 1.0
    assert result.upper == 8.0


def test_summary_multivariate_valid(multivariates):
    d_0, d_1, m_0 = multivariates
    with pytest.raises(ValueError):
        d_0.summary()
    result = d_1.summary()
    assert result.__class__.__name__ == "Dirichlet"
    assert_almost_equal(result.mean, [0.25, 0.5, 0.25])
    assert_almost_equal(result.std, [0.193, 0.223, 0.193], decimal=3)
    result = m_0.summary()
    assert result.__class__.__name__ == "MvNormal"
    assert_almost_equal(result.mean, [0, 0])
    assert_almost_equal(result.std, [1, 1])


def test_eti(a_few_poissons):
    d_0, d_1, d_2 = a_few_poissons
    with pytest.raises(ValueError):
        d_0.eti()
    with pytest.raises(ValueError):
        d_1.eti()
    result = d_2.eti()
    assert result == (1.0, 9.0)


def test_cdf(a_few_poissons):
    _, d_1, d_2 = a_few_poissons
    result1 = d_1.cdf(1)
    result2 = d_2.cdf(1)
    assert round(result2, 2) == 0.06
    assert np.allclose(result1, (0.73, 0.41), 2)


def test_ppf(a_few_poissons):
    _, d_1, d_2 = a_few_poissons
    result1 = d_1.ppf(0.5)
    result2 = d_2.ppf(0.5)
    assert np.allclose(result1, (1, 2))
    assert result2 == 4.0


def test_to_pymc():
    with Model() as model:
        Gamma(1, 1).to_pymc("a", shape=(2, 2))
        Hurdle(Gamma(1, 1), psi=0.5).to_pymc("b", shape=1)
        Truncated(Gamma(1, 1), lower=1).to_pymc("c")

    assert model.basic_RVs[0].name == "a"
    assert model.basic_RVs[0].ndim == 2
    assert model.basic_RVs[1].name == "b"
    assert model.basic_RVs[1].ndim == 1
    assert model.basic_RVs[2].name == "c"
    assert model.basic_RVs[2].ndim == 0
    assert Normal(0, 1).to_pymc(shape=2).ndim == 1
    assert Censored(Normal(0, 1), lower=0).to_pymc().ndim == 0


def test_to_bambi():
    bambi_prior = Gamma(mu=2, sigma=1).to_bambi()
    assert bambi_prior.name == "Gamma"
    assert bambi_prior.args["mu"] == 2
    assert bambi_prior.args["sigma"] == 1
