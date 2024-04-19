import pytest
from numpy.testing import assert_almost_equal
import numpy as np

from preliz.internal.optimization import find_ppf

from preliz.distributions import (
    Beta,
    Exponential,
    HalfNormal,
    Laplace,
    Normal,
    StudentT,
    Weibull,
    Geometric,
    Poisson,
)


@pytest.mark.parametrize(
    "p_dist, p_params",
    [
        (Beta, {"alpha": 2, "beta": 5}),
        (Exponential, {"beta": 3.7}),
        (HalfNormal, {"sigma": 2}),
        (Laplace, {"mu": 2.5, "b": 4}),
        (Normal, {"mu": 0, "sigma": 2}),
        (StudentT, {"nu": 5, "mu": 0, "sigma": 2}),
        (Weibull, {"alpha": 5.0, "beta": 2.0}),
        (Geometric, {"p": 0.4}),
        (Poisson, {"mu": 3.5}),
    ],
)
def test_find_ppf(p_dist, p_params):
    preliz_dist = p_dist(**p_params)
    x_vals = np.linspace(0.001, 0.999, 10)
    actual_ppf = preliz_dist.ppf(x_vals)
    expected_ppf = find_ppf(preliz_dist, x_vals)
    assert_almost_equal(actual_ppf, expected_ppf, decimal=4)
