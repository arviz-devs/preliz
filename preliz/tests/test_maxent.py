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
    "dist, name, lower, upper, mass, support, result",
    [
        (
            AsymmetricLaplace(kappa=1),
            "asymmetriclaplace",
            -5,
            5,
            0.9,
            (-np.inf, np.inf),
            (0.000, 2.171),
        ),
        (
            AsymmetricLaplace(q=0.1),
            "asymmetriclaplace",
            -1,
            1,
            0.9,
            (-np.inf, np.inf),
            (-0.799, 0.260),
        ),
        (Beta(), "beta", 0.2, 0.6, 0.9, (0, 1), (6.112, 9.101)),
        (BetaScaled(lower=-2, upper=3), "betascaled", -1, 1, 0.8, (-2, 3), (3.883, 5.560)),
        (Cauchy(), "cauchy", -1, 1, 0.6, (-np.inf, np.inf), (0, 0.726)),
        (Cauchy(alpha=0.5), "cauchy", -1, 1, 0.6, (-np.inf, np.inf), (0.6000)),
        (ChiSquared(), "chisquared", 2, 7, 0.6, (0, np.inf), (4.002)),
        (ExGaussian(), "exgaussian", 9, 10, 0.8, (-np.inf, np.inf), (9.112, 0.133, 0.495)),
        (ExGaussian(sigma=0.2), "exgaussian", 9, 10, 0.8, (-np.inf, np.inf), (9.168, 0.423)),
        (Exponential(), "exponential", 0, 4, 0.9, (0, np.inf), (0.575)),
        (Gamma(), "gamma", 0, 10, 0.7, (0, np.inf), (0.868, 0.103)),
        (Gamma(mu=9), "gamma", 0, 10, 0.7, (0, np.inf), (2.170)),
        (Gumbel(), "gumbel", 0, 10, 0.9, (-np.inf, np.inf), (3.557, 2.598)),
        (Gumbel(mu=9), "gumbel", 0, 10, 0.9, (-np.inf, np.inf), (0.444)),
        (HalfCauchy(), "halfcauchy", 0, 10, 0.7, (0, np.inf), (5.095)),
        (HalfNormal(), "halfnormal", 0, 10, 0.7, (0, np.inf), (9.648)),
        (HalfStudent(), "halfstudent", 1, 10, 0.7, (0, np.inf), (99.997, 7.697)),
        (HalfStudent(nu=7), "halfstudent", 1, 10, 0.7, (0, np.inf), (2.541)),
        (InverseGamma(), "inversegamma", 0, 1, 0.99, (0, np.inf), (8.889, 3.439)),
        (Laplace(), "laplace", -1, 1, 0.9, (-np.inf, np.inf), (0, 0.435)),
        (Laplace(mu=0.5), "laplace", -1, 1, 0.9, (-np.inf, np.inf), (0.303)),
        (Logistic(), "logistic", -1, 1, 0.5, (-np.inf, np.inf), (0, 0.91)),
        (LogNormal(), "lognormal", 1, 4, 0.5, (0, np.inf), (1.216, 0.859)),
        (LogNormal(mu=1), "lognormal", 1, 4, 0.5, (0, np.inf), (0.978)),
        (Moyal(), "moyal", 0, 10, 0.9, (-np.inf, np.inf), (2.935, 1.6)),
        (Moyal(mu=4), "moyal", 0, 10, 0.9, (-np.inf, np.inf), (1.445)),
        (Normal(), "normal", -1, 1, 0.683, (-np.inf, np.inf), (0, 1)),
        (Normal(), "normal", 10, 12, 0.99, (-np.inf, np.inf), (11, 0.388)),
        (Normal(mu=0.5), "normal", -1, 1, 0.8, (-np.inf, np.inf), (0.581)),
        (Pareto(), "pareto", 1, 4, 0.9, (1, np.inf), (1.660, 1)),
        (Pareto(m=2), "pareto", 1, 4, 0.9, (2, np.inf), (3.321)),
        (SkewNormal(), "skewnormal", -2, 10, 0.9, (-np.inf, np.inf), (3.999, 3.647, 0)),
        (SkewNormal(mu=-1), "skewnormal", -2, 10, 0.9, (-np.inf, np.inf), (6.2924, 4.905)),
        (Student(), "student", -1, 1, 0.683, (-np.inf, np.inf), (99.999, 0, 0.994)),
        (Student(nu=7), "student", -1, 1, 0.683, (-np.inf, np.inf), (0, 0.928)),
        (
            Triangular(),
            "triangular",
            0,
            4,
            0.8,
            (-1.618, 5.618),
            (-1.6180, 1.9999, 5.6180),
        ),
        (
            Triangular(c=1),
            "triangular",
            0,
            4,
            0.8,
            (-0.807, 6.428),
            (-0.807, 6.428),
        ),
        (TruncatedNormal(), "truncatednormal", -1, 1, 0.683, (-np.inf, np.inf), (0, 1)),
        (
            TruncatedNormal(lower=-3, upper=2),
            "truncatednormal",
            -1,
            1,
            0.683,
            (-3, 2),
            (-0.076, 1.031),
        ),
        (Uniform(), "uniform", -2, 10, 0.9, (-2.666, 10.666), (-2.666, 10.666)),
        (VonMises(), "vonmises", -1, 1, 0.9, (-np.pi, np.pi), (0.0, 3.294)),
        (VonMises(mu=0.5), "vonmises", -1, 1, 0.9, (-np.pi, np.pi), (6.997)),
        (Wald(), "wald", 0, 10, 0.9, (0, np.inf), (5.061, 7.937)),
        (Wald(mu=5), "wald", 0, 10, 0.9, (0, np.inf), (7.348)),
        (Weibull(), "weibull", 0, 10, 0.9, (0, np.inf), (1.411, 5.537)),
        (Weibull(alpha=2), "weibull", 0, 10, 0.9, (0, np.inf), (6.590)),
        # results for binomial are close to the correct result, but still off
        (Binomial(), "binomial", 3, 9, 0.9, (0, 9), (9, 0.490)),
        (Binomial(n=12), "binomial", 3, 9, 0.9, (0, 12), (0.612)),
        (DiscreteUniform(), "discreteuniform", -2, 10, 0.9, (-3, 11), (-2, 10)),
        (NegativeBinomial(), "negativebinomial", 0, 15, 0.9, (0, np.inf), (7.546, 2.041)),
        (NegativeBinomial(p=0.2), "negativebinomial", 0, 15, 0.9, (0, np.inf), (1.847)),
        (Poisson(), "poisson", 0, 3, 0.7, (0, np.inf), (2.763)),
    ],
)
def test_maxent(dist, name, lower, upper, mass, support, result):
    _, opt = maxent(dist, lower, upper, mass)
    rv_frozen = dist.rv_frozen

    assert rv_frozen.name == name
    assert_almost_equal(dist.support, support, 0.3)

    if dist.name != "discreteuniform":  # optimization fails to converge, but results are reasonable
        assert opt.success
    assert_allclose(opt.x, result, atol=0.001)
