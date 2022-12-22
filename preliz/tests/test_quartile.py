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
    HalfStudent,
    InverseGamma,
    Laplace,
    Logistic,
    LogNormal,
    Moyal,
    Normal,
    Pareto,
    Student,
    Triangular,
    TruncatedNormal,
    Uniform,
    VonMises,
    Wald,
    Weibull,
    # DiscreteUniform,
    NegativeBinomial,
    Poisson,
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
        (HalfStudent(), 0.5, 1, 2, (2.393, 1.311)),
        (InverseGamma(), 0.2, 0.3, 0.4, (3.881, 1.019)),
        (Laplace(), -1, 0, 1, (0, 1.442)),
        (Logistic(), -1, 0, 1, (0, 0.910)),
        (LogNormal(), 0.5, 1, 2, (0, 1.027)),
        (Moyal(), 0.5, 1, 2, (0.620, 0.567)),
        (Normal(), -1, 0, 1, (0, 1.482)),
        (Pareto(), 0.5, 1, 4, (0.541, 0.289)),
        pytest.param(
            Student(),
            -1,
            0,
            1,
            (84576.43, 0, 1.482),
            marks=pytest.mark.skipif(
                sys.version_info >= (3, 8), reason="third party implementations details"
            ),
        ),
        (Student(nu=4), -1, 0, 1, (0, 1.350)),
        (Triangular(), 0, 1, 2, (-2.414, 1.0, 4.414)),
        (TruncatedNormal(), -1, 0, 1, (0, 1.482)),
        (Uniform(), -1, 0, 1, (-2, 2)),
        (VonMises(), -1, 0, 1, (0, 0.656)),
        (Wald(), 0.5, 1, 2, (1.698, 1.109)),
        (Weibull(), 0.5, 1, 2, (1.109, 1.456)),
        # (DiscreteUniform(), -1, 0, 1, (-5, 5)), # the mass is 0.27 instead of 0.5
        (NegativeBinomial(), 3, 5, 10, (7.283, 2.167)),
        (Poisson(), 4, 5, 6, (5.641)),
    ],
)
def test_quartile(distribution, q1, q2, q3, result):
    _, opt = quartile(distribution, q1, q2, q3)

    assert_allclose(opt.x, result, atol=0.01)
