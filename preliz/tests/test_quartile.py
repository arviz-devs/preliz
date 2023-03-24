import sys
import pytest

from numpy.testing import assert_allclose

from preliz import quartile
from preliz.distributions import (
    AsymmetricLaplace,
    Beta,
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
    StudentT,
    Triangular,
    TruncatedNormal,
    Uniform,
    VonMises,
    Wald,
    Weibull,
    # Bernoulli, quartile is not useful for Bernoulli distribution as we only have two states
    BetaBinomial,
    # DiscreteUniform,
    Geometric,
    HyperGeometric,
    NegativeBinomial,
    Poisson,
    ZeroInflatedBinomial,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)


@pytest.mark.parametrize(
    "distribution, q1, q2, q3, result",
    [
        (AsymmetricLaplace(), -1, 1, 3, (1.0, 1.0, 2.885)),
        (Beta(), 0.3, 0.5, 0.7, (1.528, 1.528)),
        (Cauchy(), -1, 0, 1, (0, 1)),
        (ChiSquared(), 2, 4, 5.5, (4.329)),
        (ExGaussian(), 8, 9, 10, (8.996, 1.482, 0.003)),
        (ExGaussian(mu=8.5), 8, 9, 10, (1.401, 0.513)),
        (Exponential(), 0.5, 1, 2.5, (0.611)),
        (Gamma(), 0.5, 1, 2.5, (0.894, 0.523)),
        (Gumbel(), 0.5, 1, 2.5, (0.751, 1.265)),
        (HalfCauchy(), 0.5, 1, 3, (1.105)),
        (HalfNormal(), 0.5, 1, 2, (1.613)),
        (HalfStudentT(), 0.5, 1, 2, (2.393, 1.311)),
        (InverseGamma(), 0.2, 0.3, 0.4, (3.881, 1.019)),
        (Kumaraswamy(), 0.2, 0.3, 0.4, (2.199, 9.598)),
        (Laplace(), -1, 0, 1, (0, 1.442)),
        (Logistic(), -1, 0, 1, (0, 0.910)),
        (LogNormal(), 0.5, 1, 2, (0, 1.027)),
        (LogitNormal(), 0.3, 0.45, 0.6, (-0.212, 0.929)),
        (Moyal(), 0.5, 1, 2, (0.620, 0.567)),
        (Normal(), -1, 0, 1, (0, 1.482)),
        (Pareto(), 0.5, 1, 4, (0.541, 0.289)),
        (Rice(), 2, 4, 6, (0.03566, 3.395)),
        pytest.param(
            StudentT(),
            -1,
            0,
            1,
            (84576.43, 0, 1.482),
            marks=pytest.mark.skipif(
                sys.version_info >= (3, 8), reason="third party implementations details"
            ),
        ),
        (StudentT(nu=4), -1, 0, 1, (0, 1.350)),
        (Triangular(), 0, 1, 2, (-2.414, 1.0, 4.414)),
        (TruncatedNormal(), -1, 0, 1, (0, 1.482)),
        (Uniform(), -1, 0, 1, (-2, 2)),
        (VonMises(), -1, 0, 1, (0, 0.656)),
        (Wald(), 0.5, 1, 2, (1.698, 1.109)),
        (Weibull(), 0.5, 1, 2, (1.109, 1.456)),
        (BetaBinomial(), 2, 5, 8, (1.59, 4.49, 23)),
        # (DiscreteUniform(), -1, 0, 1, (-5, 5)), # the mass is 0.27 instead of 0.5
        (Geometric(), 2, 4, 6, (0.17)),
        (HyperGeometric(), 2, 3, 4, (75, 15, 15)),
        (NegativeBinomial(), 3, 5, 10, (7.283, 2.167)),
        (Poisson(), 4, 5, 6, (5.641)),
        (ZeroInflatedBinomial(), 3, 4, 7, (0.726, 17.001, 0.375)),
        (ZeroInflatedBinomial(psi=0.7), 2, 4, 6, (16, 0.362)),
        (ZeroInflatedNegativeBinomial(), 2, 4, 6, (0.87, 5.24, 17.49)),
        (ZeroInflatedNegativeBinomial(psi=0.9), 2, 4, 6, (5.16, 11.32)),
        (ZeroInflatedPoisson(), 4, 5, 6, (1, 5.641)),
        (ZeroInflatedPoisson(psi=0.8), 2, 4, 6, (5.475)),
    ],
)
def test_quartile(distribution, q1, q2, q3, result):
    _, opt = quartile(distribution, q1, q2, q3)

    assert_allclose(opt.x, result, atol=0.01)
