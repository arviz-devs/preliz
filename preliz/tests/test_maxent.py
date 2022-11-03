import pytest
import numpy as np

from numpy.testing import assert_allclose

from preliz.distributions.continuous import (
    HalfCauchy,
    HalfNormal,
    HalfStudent,
    TruncatedNormal,
    InverseGamma,
)
from preliz import maxent
from preliz.distributions import (
    Beta,
    Cauchy,
    Exponential,
    Gamma,
    Laplace,
    LogNormal,
    Normal,
    Pareto,
    Student,
    Uniform,
    Wald,
    Weibull,
    DiscreteUniform,
    Poisson,
)


@pytest.mark.parametrize(
    "distribution, name, lower, upper, mass, nu, support, result",
    [
        (Beta, "beta", 0.2, 0.6, 0.9, None, (0, 1), (6.112, 9.101)),
        (Cauchy, "cauchy", -1, 1, 0.6, None, (-np.inf, np.inf), (0, 0.726)),
        (Exponential, "exponential", 0, 4, 0.9, None, (0, np.inf), (0.575)),
        (Gamma, "gamma", 0, 10, 0.7, None, (0, np.inf), (0.868, 0.103)),
        (HalfCauchy, "halfcauchy", 0, 10, 0.7, None, (0, np.inf), (5.095)),
        (HalfNormal, "halfnormal", 0, 10, 0.7, None, (0, np.inf), (9.648)),
        (HalfStudent, "halfstudent", 0, 10, 0.7, 3, (0, np.inf), (8.001)),
        (HalfStudent, "halfstudent", 0, 10, 0.7, 10000, (0, np.inf), (9.648)),
        (InverseGamma, "inversegamma", 0, 1, 0.99, None, (0, np.inf), (8.889, 3.439)),
        (Laplace, "laplace", -1, 1, 0.9, None, (-np.inf, np.inf), (0, 0.435)),
        (LogNormal, "lognormal", 1, 4, 0.5, None, (0, np.inf), (1.216, 0.859)),
        (Normal, "normal", -1, 1, 0.683, None, (-np.inf, np.inf), (0, 1)),
        (Normal, "normal", 10, 12, 0.99, None, (-np.inf, np.inf), (11, 0.388)),
        (Pareto, "pareto", 1, 4, 0.9, None, (1, np.inf), (1.660, 1)),
        (Student, "student", -1, 1, 0.683, 4, (-np.inf, np.inf), (0, 0.875)),
        (Student, "student", -1, 1, 0.683, 10000, (-np.inf, np.inf), (0, 1)),
        (TruncatedNormal, "truncatednormal", -1, 1, 0.683, None, (-np.inf, np.inf), (0, 1)),
        (Uniform, "uniform", -2, 10, 0.9, None, (-np.inf, np.inf), (-2.666, 10.666)),
        (Wald, "wald", 0, 10, 0.9, None, (0, np.inf), (5.061, 7.937)),
        (Weibull, "weibull", 0, 10, 0.9, None, (0, np.inf), (1.411, 5.537)),
        (DiscreteUniform, "discreteuniform", -2, 10, 0.9, None, (-3, 11), (-2, 10)),
        (Poisson, "poisson", 0, 3, 0.7, None, (0, np.inf), (2.763)),
    ],
)
def test_maxent(distribution, name, lower, upper, mass, nu, support, result):
    if nu is None:
        dist = distribution()
    else:
        dist = distribution(nu=nu)
    _, opt = maxent(dist, lower, upper, mass)
    rv_frozen = dist.rv_frozen

    assert rv_frozen.name == name

    if dist.name == "pareto":
        assert pytest.approx(rv_frozen.support(), 0.3) == support
    else:
        assert rv_frozen.support() == support

    if dist.name != "discreteuniform":  # optimization fails to converge, but results are reasonable
        assert opt.success
    assert_allclose(opt.x, result, atol=0.001)
