# pylint: disable=no-member
from numpy.testing import assert_almost_equal
import numpy as np

from scipy import special as sc_special
from preliz.internal import special as pz_special


def test_erf():
    x = np.linspace(-2, 2, 100)
    assert_almost_equal(sc_special.erf(x), pz_special.erf(x))


def test_inv_erf():
    x = np.linspace(-0.9, 0.9, 100)
    assert_almost_equal(sc_special.erfinv(x), pz_special.erfinv(x))


def test_beta():
    a = np.linspace(0, 10, 100)
    b = np.linspace(0, 10, 100)
    assert_almost_equal(sc_special.beta(a, b), pz_special.beta(a, b))


def test_betaln():
    a = np.linspace(0, 10, 100)
    b = np.linspace(0, 10, 100)
    assert_almost_equal(sc_special.betaln(a, b), pz_special.betaln(a, b))


def test_betainc():
    x = np.linspace(0, 1, 100)
    a = np.linspace(0, 10, 100)
    b = np.linspace(0, 10, 100)
    assert_almost_equal(sc_special.betainc(a, b, x), pz_special.betainc(a, b, x))


def test_betaincinv():
    x = np.linspace(0, 1, 100)
    a = np.linspace(0, 10, 100)
    b = np.linspace(0, 10, 100)
    # in scipy < 1.12 this matches to at least 1e-7
    # in scipy >= 1.12 this matches to 1e-5
    assert_almost_equal(sc_special.betaincinv(a, b, x), pz_special.betaincinv(a, b, x))


def test_gammaln():
    x = np.linspace(0.1, 10, 100)
    assert_almost_equal(sc_special.gammaln(x), pz_special.gammaln(x))


def test_gamma():
    x = np.linspace(0.1, 10, 100)
    assert_almost_equal(sc_special.gamma(x), pz_special.gamma(x))


def test_digamma():
    x = np.linspace(0.1, 10, 100)
    assert_almost_equal(sc_special.digamma(x), pz_special.digamma(x))


def test_logit():
    x = np.linspace(-0.1, 1.1, 100)
    assert_almost_equal(sc_special.logit(x), pz_special.logit(x))


def test_expit():
    x = np.linspace(-20, 10, 500)
    assert_almost_equal(sc_special.expit(x), pz_special.expit(x))


def test_xlogy():
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 10)
    assert_almost_equal(sc_special.xlogy(x, y), pz_special.xlogy(x, y))


def test_xlog1py():
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 10)
    assert_almost_equal(sc_special.xlog1py(x, y), pz_special.xlog1py(x, y))


def test_xlogx():
    x = np.linspace(0.0, 10, 10)
    assert_almost_equal(pz_special.xlogy(x, x), pz_special.xlogx(x))
