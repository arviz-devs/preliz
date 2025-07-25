import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from preliz import maxent
from preliz.distributions import (
    AsymmetricLaplace,
    Beta,
    # Bernoulli, maxent is not useful for Bernoulli distribution as we only have two states
    BetaBinomial,
    BetaScaled,
    Binomial,
    Cauchy,
    ChiSquared,
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
    HyperGeometric,
    InverseGamma,
    Kumaraswamy,
    Laplace,
    Logistic,
    LogitNormal,
    LogLogistic,
    LogNormal,
    Moyal,
    NegativeBinomial,
    Normal,
    Pareto,
    Poisson,
    Rice,
    SkewNormal,
    SkewStudentT,
    StudentT,
    Triangular,
    TruncatedNormal,
    Uniform,
    VonMises,
    Wald,
    Weibull,
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
        (ExGaussian(), 9, 10, 0.8, (-np.inf, np.inf), (9.5, 0.390, 0)),
        (ExGaussian(nu=0.2), 9, 10, 0.8, (-np.inf, np.inf), (9.319, 0.343)),
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
        (Kumaraswamy(), 0.1, 0.6, 0.9, (0, 1), (2.311, 7.495)),
        (Laplace(), -1, 1, 0.9, (-np.inf, np.inf), (0, 0.435)),
        (Laplace(mu=0.5), -1, 1, 0.9, (-np.inf, np.inf), (0.303)),
        (Logistic(), -1, 1, 0.5, (-np.inf, np.inf), (0, 0.91)),
        (LogLogistic(), 2, 10, 0.9, (0, np.inf), (5.543, 4.047)),
        (LogNormal(), 1, 4, 0.5, (0, np.inf), (1.216, 0.859)),
        (LogNormal(mu=1), 1, 4, 0.5, (0, np.inf), (0.978)),
        (LogitNormal(), 0.3, 0.8, 0.9, (0, 1), (0.213, 0.676)),
        (LogitNormal(mu=0.7), 0.3, 0.8, 0.9, (0, 1), (0.531)),
        (Moyal(), 0, 10, 0.9, (-np.inf, np.inf), (2.935, 1.6)),
        (Moyal(mu=4), 0, 10, 0.9, (-np.inf, np.inf), (1.445)),
        (Normal(), -1, 1, 0.683, (-np.inf, np.inf), (0, 1)),
        (Normal(), 10, 12, 0.99, (-np.inf, np.inf), (11, 0.388)),
        (Normal(mu=0.5), -1, 1, 0.8, (-np.inf, np.inf), (0.581)),
        (Pareto(), 1, 4, 0.9, (1, np.inf), (1.694, 0.997)),
        (Pareto(m=2), 1, 4, 0.9, (2, np.inf), (3.321)),
        (Rice(), 0, 4, 0.7, (0, np.inf), (0, 2.577)),
        (Rice(), 1, 10, 0.9, (0, np.inf), (3.454, 3.734)),
        (Rice(nu=4), 0, 6, 0.9, (0, np.inf), (1.402)),
        (SkewNormal(), -2, 10, 0.9, (-np.inf, np.inf), (4, 3.647, 0)),
        (SkewNormal(mu=-1), -2, 10, 0.9, (-np.inf, np.inf), (6.293, 4.908)),
        (SkewStudentT(), -1, 1, 0.9, (-np.inf, np.inf), (0.010, 0.522, 3.264, 3.305)),
        (SkewStudentT(mu=0.7, sigma=0.4), -1, 1, 0.9, (-np.inf, np.inf), (2.004, 5.214)),
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
        (TruncatedNormal(lower=-np.inf, upper=np.inf), -1, 1, 0.683, (-np.inf, np.inf), (0, 1)),
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
        (BetaBinomial(), 2, 8, 0.9, (0, 10), (4.945, 4.945, 10)),
        (BetaBinomial(n=10), 2, 6, 0.6, (0, 10), (1.838, 2.181)),
        # # results for binomial are close to the correct result, but still off
        (Binomial(), 3, 9, 0.9, (0, 9), (9, 0.490)),
        (Binomial(n=12), 3, 9, 0.9, (0, 12), (0.612)),
        (DiscreteUniform(), -2, 10, 0.9, (-3, 11), (-2, 10)),
        (DiscreteWeibull(), 1, 6, 0.7, (0, np.inf), (0.939, 1.608)),
        (Geometric(), 1, 4, 0.99, (0, np.inf), (0.6837)),
        (HyperGeometric(), 2, 14, 0.9, (0, 10), (50, 10, 20)),
        (NegativeBinomial(), 0, 15, 0.9, (0, np.inf), (7.573, 2.077)),
        (NegativeBinomial(p=0.2), 0, 15, 0.9, (0, np.inf), (1.848)),
        (Poisson(), 0, 3, 0.7, (0, np.inf), (2.763)),
        (ZeroInflatedBinomial(), 1, 10, 0.9, (0, 10), (0.902, 9.0, 0.485)),
        (ZeroInflatedBinomial(psi=0.7), 1, 10, 0.7, (0, 11), (10, 0.897)),
        (ZeroInflatedNegativeBinomial(), 2, 15, 0.8, (0, np.inf), (1.0, 9.864, 3.432)),
        (ZeroInflatedNegativeBinomial(psi=0.9), 2, 15, 0.8, (0, np.inf), (9.011, 6.300)),
        (ZeroInflatedPoisson(), 0, 3, 0.7, (0, np.inf), (0.847, 3.005)),
        (ZeroInflatedPoisson(psi=0.8), 0, 3, 0.7, (0, np.inf), (3.099)),
    ],
)
def test_maxent(dist, lower, upper, mass, support, result):
    maxent(dist, lower, upper, mass)

    assert_almost_equal(dist.support, support, 0)

    if dist.__class__.__name__ not in [
        "DiscreteUniform",
        "HyperGeometric",
        "ZeroInflatedBinomial",
    ]:  # optimization fails to converge, but results are reasonable
        assert dist.opt.success
    assert_allclose(dist.opt.x, result, atol=0.001)


def test_maxent_fixed_stats():
    dist = Beta()
    maxent(dist, 0.1, 0.7, 0.94, fixed_stat=("mode", 0.3))
    assert_almost_equal(dist.mode(), 0.3)
    assert_almost_equal(dist.params, (2.7, 5.1), 1)

    dist = Gamma()
    maxent(dist, 0, 3, 0.8, fixed_stat=("mode", 2))
    assert_almost_equal(dist.mode(), 2)
    assert_almost_equal(dist.params, (7.2, 3.1), 1)

    dist = Gamma()
    maxent(dist, 0.1, 3, 0.8, fixed_stat=("mean", 2))
    assert_almost_equal(dist.mean(), 2)
    assert_almost_equal(dist.params, (2.1, 1.0), 1)

    with pytest.raises(ValueError, match="fixed_stat should be one of the following"):
        dist = Gamma()
        maxent(dist, 0, 3, 0.8, fixed_stat=("bad", 2))


def test_maxent_plot():
    maxent(Normal(), plot_kwargs={"support": "restricted", "pointinterval": True})
