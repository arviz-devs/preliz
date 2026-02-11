import numpy as np
from numpy.testing import assert_almost_equal
from scipy import special as sc_special

from preliz.internal import special as pz_special


def test_gammaln():
    x = np.linspace(0.1, 10, 100)
    assert_almost_equal(sc_special.gammaln(x), pz_special.gammaln(x))


def test_gamma():
    x = np.linspace(0.1, 10, 100)
    assert_almost_equal(sc_special.gamma(x), pz_special.gamma(x))


def test_logit():
    x = np.linspace(-0.1, 1.1, 100)
    assert_almost_equal(sc_special.logit(x), pz_special.logit(x))


def test_expit():
    x = np.linspace(-20, 10, 500)
    assert_almost_equal(sc_special.expit(x), pz_special.expit(x))
