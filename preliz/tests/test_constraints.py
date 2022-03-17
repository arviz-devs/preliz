import pytest
from preliz import constraints
import numpy as np
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    "name, name_scipy, lower, upper, mass, extra, support, result",
    [
        ("normal", "norm", -1, 1, 0.683, None, (-np.inf, np.inf), (0, 1)),
        ("normal", "norm", 10, 12, 0.99, None, (-np.inf, np.inf), (11, 0.388)),
        ("beta", "beta", 0.2, 0.6, 0.9, None, (0, 1), (6.082, 9.110)),
        ("gamma", "gamma", 0, 10, 0.7, None, (0, np.inf), (1.696, 4.822)),
        ("lognormal", "lognorm", 1, 4, 0.5, None, (0, np.inf), (0.601, 1.023)),
        ("exponential", "expon", 0, 4, 0.9, None, (0, np.inf), (1.737, 1.215)),
        ("student", "t", -1, 1, 0.683, 4, (-np.inf, np.inf), (0, 0.875)),
        ("student", "t", -1, 1, 0.683, 10000, (-np.inf, np.inf), (0, 1)),
    ],
)
def test_constraints(name, name_scipy, lower, upper, mass, extra, support, result):
    _, rv_frozen, opt = constraints(name, lower, upper, mass, extra)

    assert rv_frozen.name == name_scipy
    assert rv_frozen.support() == support
    assert opt.success
    assert_allclose(opt.x, result, atol=0.001)
