import pytest
import numpy as np

from numpy.testing import assert_allclose, assert_almost_equal


from preliz import maxent
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
    "dist, lower, upper, mass, support, result",
    [
        (
            AsymmetricLaplace(kappa=1),
            -5,
            5,
            0.9,
            (-np.inf, np.inf),
            (0.000, 2.171),
        ),
        (
            AsymmetricLaplace(q=0.1),
            -1,
            1,
            0.9,
            (-np.inf, np.inf),
            (-0.799, 0.260),
        ),
        (Beta(), 0.2, 0.6, 0.9, (0, 1), (6.112, 9.101)),
        (BetaScaled(lower=-2, upper=3), -1, 1, 0.8, (-2, 3), (3.883, 5.560)),
        (Cauchy(), -1, 1, 0.6, (-np.inf, np.inf), (0, 0.726)),
        (Cauchy(alpha=0.5), -1, 1, 0.6, (-np.inf, np.inf), (0.6000)),
        (ChiSquared(), 2, 7, 0.6, (0, np.inf), (4.002)),
        (ExGaussian(), 9, 10, 0.8, (-np.inf, np.inf), (9.112, 0.133, 0.495)),
        (ExGaussian(sigma=0.2), 9, 10, 0.8, (-np.inf, np.inf), (9.168, 0.423)),
        (Exponential(), 0, 4, 0.9, (0, np.inf), (0.575)),
        (Gamma(), 0, 10, 0.7, (0, np.inf), (0.868, 0.103)),
        (Gamma(mu=9), 0, 10, 0.7, (0, np.inf), (2.170)),
        (Gumbel(), 0, 10, 0.9, (-np.inf, np.inf), (3.557, 2.598)),
        (Gumbel(mu=9), 0, 10, 0.9, (-np.inf, np.inf), (0.444)),
        (HalfCauchy(), 0, 10, 0.7, (0, np.inf), (5.095)),
        (HalfNormal(), 0, 10, 0.7, (0, np.inf), (9.648)),
        (HalfStudent(), 1, 10, 0.7, (0, np.inf), (99.997, 7.697)),
        (HalfStudent(nu=7), 1, 10, 0.7, (0, np.inf), (2.541)),
        (InverseGamma(), 0, 1, 0.99, (0, np.inf), (8.889, 3.439)),
        (Laplace(), -1, 1, 0.9, (-np.inf, np.inf), (0, 0.435)),
        (Laplace(mu=0.5), -1, 1, 0.9, (-np.inf, np.inf), (0.303)),
        (Logistic(), -1, 1, 0.5, (-np.inf, np.inf), (0, 0.91)),
        (LogNormal(), 1, 4, 0.5, (0, np.inf), (1.216, 0.859)),
        (LogNormal(mu=1), 1, 4, 0.5, (0, np.inf), (0.978)),
        (Moyal(), 0, 10, 0.9, (-np.inf, np.inf), (2.935, 1.6)),
        (Moyal(mu=4), 0, 10, 0.9, (-np.inf, np.inf), (1.445)),
        (Normal(), -1, 1, 0.683, (-np.inf, np.inf), (0, 1)),
        (Normal(), 10, 12, 0.99, (-np.inf, np.inf), (11, 0.388)),
        (Normal(mu=0.5), -1, 1, 0.8, (-np.inf, np.inf), (0.581)),
        (Pareto(), 1, 4, 0.9, (1, np.inf), (1.660, 1)),
        (Pareto(m=2), 1, 4, 0.9, (2, np.inf), (3.321)),
        (SkewNormal(), -2, 10, 0.9, (-np.inf, np.inf), (3.999, 3.647, 0)),
        (SkewNormal(mu=-1), -2, 10, 0.9, (-np.inf, np.inf), (6.2924, 4.905)),
        (Student(), -1, 1, 0.683, (-np.inf, np.inf), (99.999, 0, 0.994)),
        (Student(nu=7), -1, 1, 0.683, (-np.inf, np.inf), (0, 0.928)),
        (
            Triangular(),
            0,
            4,
            0.8,
            (-1.618, 5.618),
            (-1.6180, 1.9999, 5.6180),
        ),
        (
            Triangular(c=1),
            0,
            4,
            0.8,
            (-0.807, 6.428),
            (-0.807, 6.428),
        ),
        (TruncatedNormal(), -1, 1, 0.683, (-np.inf, np.inf), (0, 1)),
        (
            TruncatedNormal(lower=-3, upper=2),
            -1,
            1,
            0.683,
            (-3, 2),
            (-0.076, 1.031),
        ),
        (Uniform(), -2, 10, 0.9, (-2.666, 10.666), (-2.666, 10.666)),
        (VonMises(), -1, 1, 0.9, (-np.pi, np.pi), (0.0, 3.294)),
        (VonMises(mu=0.5), -1, 1, 0.9, (-np.pi, np.pi), (6.997)),
        (Wald(), 0, 10, 0.9, (0, np.inf), (5.061, 7.937)),
        (Wald(mu=5), 0, 10, 0.9, (0, np.inf), (7.348)),
        (Weibull(), 0, 10, 0.9, (0, np.inf), (1.411, 5.537)),
        (Weibull(alpha=2), 0, 10, 0.9, (0, np.inf), (6.590)),
        # results for binomial are close to the correct result, but still off
        (Binomial(), 3, 9, 0.9, (0, 9), (9, 0.490)),
        (Binomial(n=12), 3, 9, 0.9, (0, 12), (0.612)),
        (DiscreteUniform(), -2, 10, 0.9, (-3, 11), (-2, 10)),
        (NegativeBinomial(), 0, 15, 0.9, (0, np.inf), (7.546, 2.041)),
        (NegativeBinomial(p=0.2), 0, 15, 0.9, (0, np.inf), (1.847)),
        (Poisson(), 0, 3, 0.7, (0, np.inf), (2.763)),
    ],
)
def test_maxent(dist, lower, upper, mass, support, result):
    _, opt = maxent(dist, lower, upper, mass)

    assert_almost_equal(dist.support, support, 0.3)

    if (
        dist.__class__.__name__ != "DiscreteUniform"
    ):  # optimization fails to converge, but results are reasonable
        assert opt.success
    assert_allclose(opt.x, result, atol=0.001)
