import pytest
import numpy as np
from numpy.testing import assert_allclose

from preliz import constraints
from preliz.distributions import Normal, Beta, Gamma, LogNormal, Exponential, Student


@pytest.mark.parametrize(
    "distribution, name, lower, upper, mass, nu, support, result",
    [
        (Normal, "normal", -1, 1, 0.683, None, (-np.inf, np.inf), (0, 1)),
        (Normal, "normal", 10, 12, 0.99, None, (-np.inf, np.inf), (11, 0.388)),
        (Beta, "beta", 0.2, 0.6, 0.9, None, (0, 1), (6.082, 9.110)),
        (Gamma, "gamma", 0, 10, 0.7, None, (0, np.inf), (1.696, 4.822)),
        (LogNormal, "lognormal", 1, 4, 0.5, None, (0, np.inf), (0.601, 1.023)),
        (Exponential, "exponential", 0, 4, 0.9, None, (0, np.inf), (0.575)),
        (Student, "student", -1, 1, 0.683, 4, (-np.inf, np.inf), (0, 0.875)),
        (Student, "student", -1, 1, 0.683, 10000, (-np.inf, np.inf), (0, 1)),
    ],
)
def test_constraints(distribution, name, lower, upper, mass, nu, support, result):
    if nu is None:
        dist = distribution()
    else:
        dist = distribution(nu=nu)
    _ = constraints(dist, lower, upper, mass)
    opt = dist.opt
    rv_frozen = dist.rv_frozen

    assert rv_frozen.name == name
    assert rv_frozen.support() == support
    assert opt.success
    assert_allclose(opt.x, result, atol=0.001)
