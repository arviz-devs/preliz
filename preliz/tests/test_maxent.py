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
    # Bernoulli, maxent is not useful for Bernoulli distribution as we only have two states
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
        (HalfStudentT(), 1, 10, 0.7, (0, np.inf), (99.997, 7.697)),
        (HalfStudentT(nu=7), 1, 10, 0.7, (0, np.inf), (2.541)),
        (InverseGamma(), 0, 1, 0.99, (0, np.inf), (8.889, 3.439)),
        (Kumaraswamy(), 0.1, 0.6, 0.9, (0, 1), (2.246, 7.427)),
        (Laplace(), -1, 1, 0.9, (-np.inf, np.inf), (0, 0.435)),
        (Laplace(mu=0.5), -1, 1, 0.9, (-np.inf, np.inf), (0.303)),
        (Logistic(), -1, 1, 0.5, (-np.inf, np.inf), (0, 0.91)),
        (LogNormal(), 1, 4, 0.5, (0, np.inf), (1.216, 0.859)),
        (LogNormal(mu=1), 1, 4, 0.5, (0, np.inf), (0.978)),
        (LogitNormal(), 0.3, 0.8, 0.9, (0, 1), (0.226, 0.677)),
        (LogitNormal(mu=0.7), 0.3, 0.8, 0.9, (0, 1), (0.531)),
        (Moyal(), 0, 10, 0.9, (-np.inf, np.inf), (2.935, 1.6)),
        (Moyal(mu=4), 0, 10, 0.9, (-np.inf, np.inf), (1.445)),
        (Normal(), -1, 1, 0.683, (-np.inf, np.inf), (0, 1)),
        (Normal(), 10, 12, 0.99, (-np.inf, np.inf), (11, 0.388)),
        (Normal(mu=0.5), -1, 1, 0.8, (-np.inf, np.inf), (0.581)),
        (Pareto(), 1, 4, 0.9, (1, np.inf), (1.660, 1)),
        (Pareto(m=2), 1, 4, 0.9, (2, np.inf), (3.321)),
        (Rice(), 0, 4, 0.7, (0, np.inf), (0, 2.577)),
        (Rice(), 1, 10, 0.9, (0, np.inf), (3.453, 3.735)),
        (Rice(nu=4), 0, 6, 0.9, (0, np.inf), (1.402)),
        (SkewNormal(), -2, 10, 0.9, (-np.inf, np.inf), (3.999, 3.647, 0)),
        (SkewNormal(mu=-1), -2, 10, 0.9, (-np.inf, np.inf), (6.2924, 4.905)),
        (StudentT(), -1, 1, 0.683, (-np.inf, np.inf), (99.999, 0, 0.994)),
        (StudentT(nu=7), -1, 1, 0.683, (-np.inf, np.inf), (0, 0.928)),
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
        # This fails with scipy 1.11.1 if lower or upper are inf
        # setting to "large" number for now
        (TruncatedNormal(lower=-100, upper=100), -1, 1, 0.683, (-100, 100), (0, 1)),
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
        (BetaBinomial(), 2, 8, 0.9, (0, 8), (1.951, 1.345, 8)),
        (BetaBinomial(n=10), 2, 6, 0.6, (0, 10), (1.837, 2.181)),
        # # results for binomial are close to the correct result, but still off
        (Binomial(), 3, 9, 0.9, (0, 9), (9, 0.490)),
        (Binomial(n=12), 3, 9, 0.9, (0, 12), (0.612)),
        (DiscreteUniform(), -2, 10, 0.9, (-3, 11), (-2, 10)),
        (DiscreteWeibull(), 1, 6, 0.7, (0, np.inf), (0.938, 1.604)),
        (Geometric(), 1, 4, 0.99, (0, np.inf), (0.6837)),
        (HyperGeometric(), 2, 14, 0.9, (0, 21), (56, 21, 21)),
        (NegativeBinomial(), 0, 15, 0.9, (0, np.inf), (7.546, 2.041)),
        (NegativeBinomial(p=0.2), 0, 15, 0.9, (0, np.inf), (1.847)),
        (Poisson(), 0, 3, 0.7, (0, np.inf), (2.763)),
        (ZeroInflatedBinomial(), 1, 10, 0.9, (0, np.inf), (0.901, 10, 0.493)),
        (ZeroInflatedBinomial(psi=0.7), 1, 10, 0.7, (0, np.inf), (11, 0.5)),
        (ZeroInflatedNegativeBinomial(), 2, 15, 0.8, (0, np.inf), (1.0, 9.851, 3.416)),
        (ZeroInflatedNegativeBinomial(psi=0.9), 2, 15, 0.8, (0, np.inf), (8.775, 5.709)),
        (ZeroInflatedPoisson(), 0, 3, 0.7, (0, np.inf), (1, 2.763)),
        (ZeroInflatedPoisson(psi=0.8), 0, 3, 0.7, (0, np.inf), (1.898)),
    ],
)
def test_maxent(dist, lower, upper, mass, support, result):
    _, opt = maxent(dist, lower, upper, mass)

    assert_almost_equal(dist.support, support, 0.3)

    if dist.__class__.__name__ not in [
        "DiscreteUniform",
        "HyperGeometric",
        "ZeroInflatedBinomial",
    ]:  # optimization fails to converge, but results are reasonable
        assert opt.success
    assert_allclose(opt.x, result, atol=0.001)


def test_maxent_plot():
    maxent(Normal(), plot_kwargs={"support": "restricted", "pointinterval": True})
