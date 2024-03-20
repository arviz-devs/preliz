import pytest
from numpy.testing import assert_allclose

import preliz as pz


from preliz.distributions import (
    AsymmetricLaplace,
    Beta,
    BetaScaled,
    Cauchy,
    ChiSquared,
    ExGaussian,
    Exponential,
    Gamma,
    Gumbel,
    HalfCauchy,
    HalfNormal,
    HalfStudentT,
    InverseGamma,
    Kumaraswamy,
    Laplace,
    Logistic,
    LogNormal,
    LogitNormal,
    Moyal,
    Normal,
    Pareto,
    Rice,
    SkewNormal,
    StudentT,
    Triangular,
    TruncatedNormal,
    Uniform,
    VonMises,
    Wald,
    Weibull,
    Bernoulli,
    BetaBinomial,
    Binomial,
    DiscreteUniform,
    DiscreteWeibull,
    Geometric,
    HyperGeometric,
    NegativeBinomial,
    Poisson,
    ZeroInflatedBinomial,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)


@pytest.mark.parametrize(
    "distribution, params",
    [
        (AsymmetricLaplace, (2, 3, 1)),
        (Beta, (2, 5)),
        (BetaScaled, (2, 5, -1, 4)),
        (Cauchy, (0, 1)),
        (ChiSquared, (1,)),
        (ExGaussian, (0, 1, 3)),
        (Exponential, (5,)),
        (Gamma, (2, 5)),
        (Gumbel, (0, 2)),
        (HalfCauchy, (1,)),
        (HalfNormal, (1,)),
        (HalfStudentT, (3, 1)),
        (HalfNormal, (2,)),
        (InverseGamma, (3, 5)),
        (Kumaraswamy, (2, 3)),
        (Laplace, (0, 2)),
        (Logistic, (0, 1)),
        (LogNormal, (0, 1)),
        (LogitNormal, (0, 0.5)),
        (Moyal, (0, 2)),
        (Normal, (0, 1)),
        (Pareto, (5, 1)),
        (Rice, (0, 2)),
        (SkewNormal, (0, 1, 6)),
        (StudentT, (4, 0, 1)),
        (Triangular, (0, 2, 4)),
        (TruncatedNormal, (0, 1, -1, 1)),
        (Uniform, (2, 5)),
        (VonMises, (1, 2)),
        (Wald, (2, 1)),
        (Weibull, (2, 1)),
        (Bernoulli, (0.5,)),
        (BetaBinomial, (1, 2, 10)),
        (Binomial, (5, 0.5)),
        (DiscreteUniform, (-2, 2)),
        (DiscreteWeibull, (0.9, 1.3)),
        (Geometric, (0.75,)),
        (HyperGeometric, (50, 20, 10)),
        (NegativeBinomial, (10, 0.5)),
        (Poisson, (4.2,)),
        (ZeroInflatedBinomial, (0.5, 10, 0.8)),
        (ZeroInflatedNegativeBinomial, (0.7, 8, 4)),
        (
            ZeroInflatedPoisson,
            (
                0.8,
                4.2,
            ),
        ),
    ],
)
def test_auto_recover(distribution, params):
    dist = distribution(*params)
    sample = dist.rvs(10000)
    pz.mle([distribution()], sample)
    assert_allclose(dist.params, params, atol=1)


def test_recover_right():
    dists = [Normal(), Gamma(), Poisson()]
    sample = Normal(0, 1).rvs(10000)
    pz.mle(dists, sample)
    assert dists[0].__class__.__name__ == "Normal"

    sample = Gamma(2, 10).rvs(10000)
    pz.mle(dists, sample)
    assert dists[1].__class__.__name__ == "Gamma"

    sample = Poisson(10).rvs(10000)
    pz.mle(dists, sample)
    assert dists[2].__class__.__name__ == "Poisson"
